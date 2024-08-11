def recommend_locations_hybrid1(traveler_id, combined_similarity_df, visit_matrix, n_recommendations=20):
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

    recommended_locations = visit_matrix.loc[similar_users].sum(axis=0)

    if traveler_id in visit_matrix.index:
        visited_locations = visit_matrix.loc[traveler_id]
        recommended_locations = recommended_locations[visited_locations == 0]
    else:
        recommended_locations = recommended_locations

    recommended_locations = recommended_locations.sort_values(ascending=False).head(n_recommendations)

    return recommended_locations.index.tolist()
