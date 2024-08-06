# # n_recommendations 이 추천할 장소 개수, 기본값=20

def recommend_locations_hybrid1(traveler_id, combined_similarity_df, visit_matrix, n_recommendations=20):
    if traveler_id not in combined_similarity_df.index:
        raise ValueError("Traveler ID 가 유효하지 않음.")
    
    similar_users = combined_similarity_df[traveler_id].sort_values(ascending=False)
    similar_users = similar_users.index[similar_users.index != traveler_id]
    recommended_locations = visit_matrix.loc[similar_users].sum(axis=0)
    visited_locations = visit_matrix.loc[traveler_id]
    recommended_locations = recommended_locations[visited_locations == 0]
    recommended_locations = recommended_locations.sort_values(ascending=False).head(n_recommendations)

     '''  해인님이 추가해주실 item based 추천과 결합 로직 필요 '''

    return recommended_locations.index.tolist()
