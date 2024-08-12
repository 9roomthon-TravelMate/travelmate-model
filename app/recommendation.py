from sklearn.metrics.pairwise import cosine_similarity

def recommend_locations_hybrid1(traveler_id, combined_similarity_df, visit_matrix, content_embeddings, n_recommendations=20):
    if traveler_id not in combined_similarity_df.index:
        raise ValueError("Traveler ID가 유효하지 않음.")
        
    # traveler_id 행을 가져와서 시리즈로 변환
    similar_users = combined_similarity_df.loc[traveler_id, :].sort_values(ascending=False)
    similar_users = similar_users.index[similar_users.index != traveler_id]
    
    # 필터링된 similar_users에서 visit_matrix에 존재하는 사용자만 선택
    similar_users = [user for user in similar_users if user in visit_matrix.index]
    

    # similar_users가 없다면 빈 추천 목록 반환
    if not similar_users:
        return []
    
    first_similar_user = similar_users[0]
    visited_locations = visit_matrix.loc[first_similar_user]
    visited_content_ids = int(visited_locations[visited_locations != 0].index.tolist()[0])
    recommended_locations_cf = visit_matrix.loc[similar_users].sum(axis=0)
    recommended_locations_cbf = get_content_based_recommendation(visited_content_ids, content_embeddings, top_k=n_recommendations // 2)

    if traveler_id in visit_matrix.index:
        visited_locations = visit_matrix.loc[traveler_id]
        recommended_locations_cf = recommended_locations_cf[visited_locations == 0]
    else:
        recommended_locations_cf = recommended_locations_cf

    recommended_locations_cf = recommended_locations_cf.sort_values(ascending=False).head(n_recommendations // 2)
    recommended_locations_cf = recommended_locations_cf.index.tolist()
    
    return recommended_locations_cf + recommended_locations_cbf


def get_content_based_recommendation(contentid, content_embeddings, top_k: int = 10):
    idx_map = {i: idx for i, idx in enumerate(content_embeddings.index)}
    
    query = content_embeddings[content_embeddings.index == contentid]
    others = content_embeddings[content_embeddings.index != contentid]
   
    recommendation =cosine_similarity(query, others).argsort()[:, ::-1][0, :top_k]
   
    return [idx_map[idx] for idx in recommendation]

