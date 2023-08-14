import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_data(df_traveller, df_visited):
    # 날짜 데이터 전처리 (datetime 변환)
    df_traveller['travel_start_ymd'] = pd.to_datetime(df_traveller['travel_start_ymd'])
    df_traveller['travel_end_ymd'] = pd.to_datetime(df_traveller['travel_end_ymd'])
    df_traveller['start_month'] = df_traveller['travel_start_ymd'].dt.month.astype(np.int8)

    # 카테고리형 데이터 OneHotEncoding
    categorical_features = ['gender', 'age_grp', 'start_month']
    encoder = OneHotEncoder(sparse_output=True)
    encoded_features = encoder.fit_transform(df_traveller[categorical_features])

    df_traveller_encoded = pd.DataFrame.sparse.from_spmatrix(
        encoded_features, index=df_traveller.index, columns=encoder.get_feature_names_out(categorical_features)
    ).astype(np.float32)

    df_traveller_encoded = pd.concat([df_traveller[['traveler_id']], df_traveller_encoded], axis=1).dropna()
    df_traveller_encoded.set_index('traveler_id', inplace=True)

    # 사용자 프로필 유사도 계산
    user_profile_similarity = cosine_similarity(df_traveller_encoded.to_numpy(dtype=np.float32))
    user_profile_similarity_df = pd.DataFrame(user_profile_similarity, index=df_traveller_encoded.index, columns=df_traveller_encoded.index)

    # 방문 기록 매트릭스 생성
    visit_matrix = df_visited.pivot_table(index='traveler_id', columns='content_id', aggfunc='size', fill_value=0)
    visit_matrix = visit_matrix.astype(np.int8)

    # 방문 기록 유사도 계산
    if visit_matrix.empty:
        user_visit_similarity_df = pd.DataFrame(0, index=df_traveller_encoded.index, columns=df_traveller_encoded.index)
    else:
        user_visit_similarity = cosine_similarity(visit_matrix.to_numpy(dtype=np.float32))
        user_visit_similarity_df = pd.DataFrame(user_visit_similarity, index=visit_matrix.index, columns=visit_matrix.index)
        user_visit_similarity_df = user_visit_similarity_df.reindex(index=df_traveller_encoded.index, columns=df_traveller_encoded.index, fill_value=0)

    # 유효한 사용자만 필터링
    user_profile_similarity_df = user_profile_similarity_df.reindex(visit_matrix.index, columns=visit_matrix.columns, fill_value=0)
    user_visit_similarity_df = user_visit_similarity_df.reindex(visit_matrix.index, columns=visit_matrix.columns, fill_value=0)

    # 가중치 기반 유사도 결합
    weight_profile = 0.7
    weight_visit = 0.3
    combined_similarity_df = (user_profile_similarity_df * weight_profile + user_visit_similarity_df * weight_visit).astype(np.float32)

    del user_profile_similarity, user_visit_similarity
    gc.collect()

    return df_traveller_encoded, visit_matrix, combined_similarity_df

