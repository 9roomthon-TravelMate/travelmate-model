from sklearn.metrics.pairwise import cosine_similarity

def recommend_locations_hybrid1(traveler_id, combined_similarity_df, visit_matrix, content_embeddings, n_recommendations=20):
    try:
        # 유효한 Traveler ID인지 확인
        if traveler_id not in combined_similarity_df.index:
            raise ValueError(f"Traveler ID {traveler_id}가 유효하지 않음.")
        
        print(f"Traveler ID: {traveler_id}")
        similar_users = combined_similarity_df.loc[traveler_id].sort_values(ascending=False)
        similar_users = similar_users.index[similar_users.index != traveler_id]
        print(f"Similar users: {similar_users}")
        
        if similar_users.empty:
            raise ValueError(f"No similar users found for traveler ID {traveler_id}.")
        
        # 협업 필터링: 유사 사용자들의 가중치 합산을 통한 추천
        weighted_recommendations = visit_matrix.loc[similar_users].multiply(similar_users.values, axis=0).sum(axis=0)
        recommended_locations_cf = weighted_recommendations.sort_values(ascending=False).head(n_recommendations // 2)
        recommended_locations_cf = recommended_locations_cf.index.tolist()
        print(f"Recommended locations (CF): {recommended_locations_cf}")
        
        # 콘텐츠 기반 필터링: 첫 번째 유사 사용자의 방문지를 기반으로 추천
        first_similar_user = similar_users[0]
        visited_locations = visit_matrix.loc[first_similar_user]
        visited_content_ids = visited_locations[visited_locations != 0].index.tolist()
        print(f"Visited content IDs: {visited_content_ids}")

        recommended_locations_cbf = []
        if visited_content_ids:
            try:
                recommended_locations_cbf = get_content_based_recommendation(visited_content_ids[0], content_embeddings, top_k=n_recommendations // 2)
                print(f"Recommended locations (CBF): {recommended_locations_cbf}")
            except Exception as e:
                print(f"Error during CBF recommendation: {str(e)}")
        else:
            print("No visited content IDs found for CBF.")

        return recommended_locations_cf + recommended_locations_cbf
    
    except ValueError as e:
        # Traveler ID가 유효하지 않은 경우, CBF로만 추천
        print(str(e))
        try:
            # 기본 콘텐츠 기반 추천 - 아이템 기반 추천
            all_content_ids = content_embeddings.index.tolist()
            recommended_locations_cbf = []
            if all_content_ids:
                # 가장 유사한 사용자 선택하여 해당 사용자가 방문한
                recommended_locations_cbf = get_content_based_recommendation(all_content_ids[0], content_embeddings, top_k=n_recommendations)
                print(f"Recommended locations (CBF - Item Based): {recommended_locations_cbf}")
            else:
                print("No content IDs found for item-based recommendation.")
            return recommended_locations_cbf
        except Exception as e:
            print(f"Error during CBF recommendation for item-based recommendation: {str(e)}")
            return []


def get_content_based_recommendation(contentid, content_embeddings, top_k: int = 10):

    contentid = int(contentid)
    query = content_embeddings.loc[contentid].values.reshape(1, -1)

    try:
        similarity_scores = cosine_similarity(query, content_embeddings.drop(contentid)).flatten()
    except Exception as e:
        print(f"Error in calculating similarity: {str(e)}")
        raise

    most_similar_indices = similarity_scores.argsort()[::-1][:top_k]
    recommendation = content_embeddings.drop(contentid).iloc[most_similar_indices].index.tolist()
    
    return recommendation
