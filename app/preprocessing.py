import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_data(df_traveller, df_visited):
    # 날짜 데이터 전처리
    df_traveller['travel_start_ymd'] = pd.to_datetime(df_traveller['travel_start_ymd'])
    df_traveller['travel_end_ymd'] = pd.to_datetime(df_traveller['travel_end_ymd'])
    df_traveller['start_month'] = df_traveller['travel_start_ymd'].dt.month

    # 카테고리형 데이터 인코딩
    categorical_features = ['gender', 'age_grp', 'start_month']
    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform(df_traveller[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

    # traveler_id 중복 제거 (필요한 경우)
    df_traveller = df_traveller.drop_duplicates(subset=['traveler_id'])

    # NaN 값을 포함한 행을 삭제
    df_traveller_encoded = pd.concat(
        [df_traveller[['traveler_id']], encoded_df, df_traveller.drop(columns=categorical_features + ['traveler_id', 'travel_start_ymd', 'travel_end_ymd'])],
        axis=1
    ).dropna()  # NaN 값을 포함한 행을 삭제합니다
    df_traveller_encoded.set_index('traveler_id', inplace=True)

    # 사용자 프로필 유사도 계산
    user_profile_similarity = cosine_similarity(df_traveller_encoded)
    user_profile_similarity_df = pd.DataFrame(user_profile_similarity, index=df_traveller_encoded.index, columns=df_traveller_encoded.index)

    # 방문 기록 매트릭스 생성
    visit_matrix = df_visited.pivot_table(index='traveler_id', columns='content_id', aggfunc='size', fill_value=0)

    # 방문 기록 유사도 계산
    if visit_matrix.empty:
        user_visit_similarity_df = pd.DataFrame(0, index=df_traveller_encoded.index, columns=df_traveller_encoded.index)
    else:
        user_visit_similarity = cosine_similarity(visit_matrix)
        user_visit_similarity_df = pd.DataFrame(user_visit_similarity, index=visit_matrix.index, columns=visit_matrix.index)
        # `user_visit_similarity_df`를 `df_traveller_encoded`의 인덱스를 기준으로 재인덱싱
        user_visit_similarity_df = user_visit_similarity_df.reindex(index=df_traveller_encoded.index, columns=df_traveller_encoded.index, fill_value=0)

    # 프로필 유사도와 방문 기록 유사도를 결합
    weight_profile = 0.7
    weight_visit = 0.3
    combined_similarity_df = (user_profile_similarity_df * weight_profile + user_visit_similarity_df * weight_visit)

    return df_traveller_encoded, visit_matrix, combined_similarity_df
