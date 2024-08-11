
from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
import crud
import models
from database import SessionLocal, engine
from recommendation import recommend_locations_hybrid1
from preprocessing import preprocess_data
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import pandas as pd
import os

load_dotenv()  # .env 파일 로드

models.Base.metadata.create_all(bind=engine)

content_embeddings = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global content_embeddings
    content_embeddings = pd.read_csv(os.getenv('CONTENT_EMBEDDINGS_PATH'), index_col='contentid')
    yield

app = FastAPI(lifespan=lifespan)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def generate_similarity_matrices(db: Session, region_id: int = None):
    preferences = crud.get_all_preferences(db)
    visited = crud.get_all_visited(db)
    tour_spots = crud.get_tour_spots(db, region_id)
    valid_content_ids = set(tour_spot.content_id for tour_spot in tour_spots)

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
    } for pref in preferences])

    df_visited = pd.DataFrame([{
        'traveler_id': visit.traveler_id,
        'content_id': visit.content_id
    } for visit in visited])

    return preprocess_data(df_traveller, df_visited, valid_content_ids)

# 지역 코드 param에 있으면 해당 지역으로 필터링, 지역 선택 안하면 (param에 없으면) 전체 지역 대상으로 추천


@app.get("/recommend/{traveler_id}")
def get_recommendations(traveler_id: int, region_id: int = Query(None), n_recommendations: int = 20, db: Session = Depends(get_db)):
    try:
        df_traveller_encoded, visit_matrix, combined_similarity_df = generate_similarity_matrices(db, region_id)
        recommendations = recommend_locations_hybrid1(traveler_id, combined_similarity_df, visit_matrix, n_recommendations)
        return {"traveler_id": traveler_id, "recommendations": recommendations}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
