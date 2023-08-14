
from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app import crud
from app import models
from app.database import SessionLocal, engine
from app.recommendation import recommend_locations_hybrid1
from app.preprocessing import preprocess_data
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import pandas as pd
import os
import psutil
import sys
import logging
import numpy as np
import gc

# ✅ 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv() 

models.Base.metadata.create_all(bind=engine)

CHUNK_SIZE = 5000 # 청크 크기 설정(현재 약 48000개의 데이터 존재)

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global content_embeddings
#     s3 = boto3.client('s3')
#     s3.download_file('travel-mate-model-server', 'model/visited_embedding.csv', 'visited_embedding.csv')
#     content_embeddings = pd.read_csv('visited_embedding.csv', index_col='contentid')
#     yield
#     os.remove('visited_embedding.csv')

def load_content_embeddings():
    file_path = "visited_embedding.csv"
    if not os.path.exists(file_path):
        raise RuntimeError(f"Error: {file_path} 파일을 찾을 수 없습니다.")
    
    logger.info(f"📌 {file_path} 파일을 chunk 단위({CHUNK_SIZE})로 로드합니다.")
    return pd.read_csv(file_path, index_col='contentid', chunksize=CHUNK_SIZE)


def generate_similarity_matrices(db: Session, traveler_id: int = None, region_id: int = None):
    preferences = crud.get_all_preferences(db, traveler_id=traveler_id, region_id=region_id)
    visited = crud.get_all_visited(db, region_id)

    df_traveller = pd.DataFrame([{
        'traveler_id': pref.traveler_id,
        'gender': pref.gender,
        'age_grp': pref.age_grp,
        'travel_start_ymd': pref.travel_start_ymd,
        'travel_end_ymd': pref.travel_end_ymd,
        'travel_styl_1': pref.travel_styl_1,
        'travel_styl_2': pref.travel_styl_2,
        'travel_styl_3': pref.travel_styl_3,
        'travel_styl_4': pref.travel_styl_4,
        'travel_styl_5': pref.travel_styl_5,
        'travel_styl_6': pref.travel_styl_6,
        'travel_styl_7': pref.travel_styl_7,
        'travel_companions_num': pref.travel_companions_num
    } for pref in preferences]) if preferences else pd.DataFrame()

    df_visited = pd.DataFrame([{
        'traveler_id': visit.traveler_id,
        'content_id': visit.content_id
    } for visit in visited]) if visited else pd.DataFrame()

    if df_traveller.empty or df_visited.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    return preprocess_data(df_traveller, df_visited)

@app.get("/recommend/{traveler_id}")
def get_recommendations(traveler_id: int, region_id: int = Query(None), n_recommendations: int = 20, db: Session = Depends(get_db)):
    try:
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 ** 2  
        peak_mem = mem_before  

        logger.info(f" traveler_id={traveler_id}, region_id={region_id} 추천 시작")
        logger.info(f" API 호출 전 시스템 메모리 사용량: {mem_before:.2f} MB")

        recommendations_with_scores = []

        for i, chunk in enumerate(load_content_embeddings()):
           
            mem_after_chunk_load = process.memory_info().rss / 1024 ** 2
            logger.info(f" [{i+1}번째 청크] 로드 완료 (크기: {len(chunk)}) | 메모리 사용량: {mem_after_chunk_load:.2f} MB")

            if region_id is not None:
                chunk = chunk[chunk['areacode'] == region_id]
            
            mem_after_filtering = process.memory_info().rss / 1024 ** 2
            logger.info(f"[{i+1}번째 청크] 필터링 후 크기: {len(chunk)} | 메모리 사용량: {mem_after_filtering:.2f} MB")

            if chunk.empty:
                continue  

            df_traveller_encoded, visit_matrix, combined_similarity_df = generate_similarity_matrices(db, traveler_id=traveler_id, region_id=region_id)

            if df_traveller_encoded.empty or visit_matrix.empty or combined_similarity_df.empty:
                continue  

            visit_matrix_np = visit_matrix.to_numpy(dtype=np.float32, copy=False)

            logger.info(f"[{i+1}번째 청크] traveler_id={traveler_id} 추천 수행 (최종 청크 크기: {len(chunk)})")

            chunk_recommendations = recommend_locations_hybrid1(traveler_id, combined_similarity_df, visit_matrix_np, chunk, n_recommendations)
            recommendations_with_scores.extend(chunk_recommendations)

            mem_after_chunk_process = process.memory_info().rss / 1024 ** 2
            logger.info(f"[{i+1}번째 청크] 추천 수행 후 메모리 사용량: {mem_after_chunk_process:.2f} MB")

            gc.collect()
            # mem_after_gc = process.memory_info().rss / 1024 ** 2
            # logger.info(f" [{i+1}번째 청크] GC 실행 후 메모리 사용량: {mem_after_gc:.2f} MB")

            if mem_after_chunk_process > peak_mem:
                peak_mem = mem_after_chunk_process  # ✅ 피크 메모리 갱신

        recommendations_with_scores.sort(key=lambda x: x[1], reverse=True)
        recommendations = [content_id for content_id, _ in recommendations_with_scores][:n_recommendations]

        mem_after = process.memory_info().rss / 1024 ** 2
        logger.info(f" traveler_id={traveler_id} 추천 완료!")
        logger.info(f" API 호출 후 시스템 메모리 사용량: {mem_after:.2f} MB")
        logger.info(f" 전체 처리 중 최고 메모리 사용량: {peak_mem:.2f} MB")

        if not recommendations:
            raise HTTPException(status_code=404, detail="No recommendations available")

        return {
            "traveler_id": traveler_id,
            "recommendations": recommendations,
            "system_memory_before": f"{mem_before:.2f} MB",
            "system_memory_after": f"{mem_after:.2f} MB",
            "peak_system_memory": f"{peak_mem:.2f} MB"
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
