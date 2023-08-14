from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import gc

def recommend_locations_hybrid1(traveler_id, combined_similarity_df, visit_matrix, content_embeddings, n_recommendations=20):
    try:
        # traveler_id 검증
        if traveler_id not in combined_similarity_df.index:
            raise ValueError(f"Traveler ID {traveler_id}가 유효하지 않음.")

        # NumPy로 변환 (pandas 연산보다 빠름름)
        similarity_values = combined_similarity_df.loc[traveler_id].to_numpy(dtype=np.float32)
        similar_users = combined_similarity_df.index[similarity_values.argsort()[::-1]] 

        # 자기 자신 제외
        similar_users = similar_users[similar_users != traveler_id]
        
        if similar_users.size == 0:
            raise ValueError(f"No similar users found for traveler ID {traveler_id}.")

        # 1. 방문기록 + 유저 취향 유사도 기반 추천
        visit_matrix_np = visit_matrix.to_numpy(dtype=np.float32)  # ✅ NumPy 변환
        weighted_recommendations = visit_matrix_np[similar_users].T @ similarity_values[:len(similar_users)]
        recommended_locations_cf = np.argsort(weighted_recommendations)[::-1][:n_recommendations * 2]

        recommended_locations_cf = [(int(content_id), float(weighted_recommendations[content_id])) for content_id in recommended_locations_cf]

        # 2. 아이템 유사도 기반 추천
        recommended_locations_cbf = []
        if len(similar_users) > 0:
            first_similar_user = similar_users[0]
            visited_content_ids = visit_matrix.loc[first_similar_user][visit_matrix.loc[first_similar_user] != 0].index.tolist()

            if visited_content_ids:
                recommended_locations_cbf = get_content_based_recommendation(visited_content_ids[0], content_embeddings, top_k=n_recommendations * 2)

        recommended_locations_cbf = [(int(content_id), float(score)) for content_id, score in recommended_locations_cbf]

        # 1과 2의 추천 리스트 합치고, 점수 기준 정렬
        combined_recommendations = list(dict.fromkeys(recommended_locations_cf + recommended_locations_cbf))
        combined_recommendations = sorted(combined_recommendations, key=lambda x: x[1], reverse=True)

        gc.collect() 

        return combined_recommendations[:n_recommendations]

    except ValueError:
        # 기본 콘텐츠 기반 추천 
        all_content_ids = content_embeddings.index.tolist()
        if all_content_ids:
            return [(int(content_id), float(score)) for content_id, score in get_content_based_recommendation(all_content_ids[0], content_embeddings, top_k=n_recommendations)]
        return []


def get_content_based_recommendation(contentid, content_embeddings, top_k=10):
    contentid = int(contentid)
    query = content_embeddings.loc[contentid].to_numpy(dtype=np.float32).reshape(1, -1) 

    similarity_scores = cosine_similarity(query, content_embeddings.drop(contentid).to_numpy(dtype=np.float32)).flatten()
    most_similar_indices = similarity_scores.argsort()[::-1][:top_k]

    recommended_items = content_embeddings.drop(contentid).iloc[most_similar_indices].index.tolist()
    recommended_scores = [float(score) for score in similarity_scores[most_similar_indices]]  

    return list(zip(recommended_items, recommended_scores))  