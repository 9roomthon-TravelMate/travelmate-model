import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_data(df_traveller, df_visited, valid_content_ids=None):
    # 날짜 데이터 전처리
    df_traveller['travel_start_ymd'] = pd.to_datetime(df_traveller['travel_start_ymd'])
    df_traveller['travel_end_ymd'] = pd.to_datetime(df_traveller['travel_end_ymd'])
    df_traveller['start_month'] = df_traveller['travel_start_ymd'].dt.month

    # 카테고리형 데이터 인코딩
    categorical_features = ['GENDER', 'AGE_GRP', 'START_MONTH']                        
    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform(df_traveller[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

    # 사용자 데이터 프레임 결합
    df_traveller_encoded = pd.concat(
        [df_traveller[['traveler_id']], encoded_df, df_traveller.drop(columns=categorical_features + ['traveler_id', 'travel_start_ymd', 'travel_end_ymd'])],
        axis=1
    )
    df_traveller_encoded.set_index('traveler_id', inplace=True)

    # 사용자 프로필 유사도 계산
    user_profile_similarity = cosine_similarity(df_traveller_encoded)
    user_profile_similarity_df = pd.DataFrame(user_profile_similarity, index=df_traveller_encoded.index, columns=df_traveller_encoded.index)

    # 방문 기록 매트릭스 생성
    visit_matrix = df_visited.pivot_table(index='traveler_id', columns='content_id', aggfunc='size', fill_value=0)


    # 유효한 콘텐츠 ID로 필터링
    if valid_content_ids is not None:
        filtered_columns = visit_matrix.columns.intersection(valid_content_ids)
        if filtered_columns.empty:
            print("사용자와 지역 내에 동일한 콘텐츠를 방문한 다른 사용자가 없어 방문기록 행렬은 초기화하여 사용합니다.")
        else:
            visit_matrix = visit_matrix[filtered_columns]
            print("디버깅: 필터링된 방문 기록 매트릭스")
            print(visit_matrix.head())
            print("디버깅: 필터링된 방문 기록 매트릭스 크기")
            print(visit_matrix.shape)

    # 방문 기록 유사도 계산
    if visit_matrix.empty:
        print("사용자와 동일한 콘텐츠를 방문한 다른 사용자가 없어 방문기록 행렬은 초기화하여 사용합니다.")
        user_visit_similarity_df = pd.DataFrame(0, index=df_traveller_encoded.index, columns=df_traveller_encoded.index)
    else:
        user_visit_similarity = cosine_similarity(visit_matrix)
        user_visit_similarity_df = pd.DataFrame(user_visit_similarity, index=visit_matrix.index, columns=visit_matrix.index)

    # 프로필 유사도와 방문 기록 유사도를 결합
    weight_profile = 0.7
    weight_visit = 0.3
    combined_similarity_df = (user_profile_similarity_df * weight_profile + user_visit_similarity_df * weight_visit)


    return df_traveller_encoded, visit_matrix, combined_similarity_df
