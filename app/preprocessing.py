import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_data(df_traveller, df_visited, valid_content_ids=None):
    
    # 날짜 데이터 전처리
    df_traveller['travel_start_ymd'] = pd.to_datetime(df_traveller['travel_start_ymd'])
    df_traveller['travel_end_ymd'] = pd.to_datetime(df_traveller['travel_end_ymd'])
    df_traveller['start_month'] = df_traveller['travel_start_ymd'].dt.month # 유사도 행렬 그릴 땐, 날짜 데이터에서 여행 시작 '월' 만 뽑아서 사용

    categorical_features = ['gender'] # 성별 데이터 인코딩
    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform(df_traveller[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

    df_traveller_encoded = pd.concat([df_traveller[['traveler_id']], encoded_df, df_traveller.drop(columns=categorical_features + ['traveler_id', 'travel_start_ymd', 'travel_end_ymd'])], axis=1)
    df_traveller_encoded.set_index('traveler_id', inplace=True) # 전처리 완료된 데이터 프레임 (사용자 설문 데이터)

    # 사용자 설문 데이터 기반으로 유사도 행렬 생성
    user_profile_similarity = cosine_similarity(df_traveller_encoded)
    user_profile_similarity_df = pd.DataFrame(user_profile_similarity, index=df_traveller_encoded.index, columns=df_traveller_encoded.index)

    # 방문 기록 기반 유사도 행렬 생성
    visit_matrix = df_visited.pivot_table(index='traveler_id', columns='content_id', aggfunc='size', fill_value=0)

    # 지역 코드를 선택했으면 해당 지역에 해당하는 content_id 만 선택하여 유사도 행렬 생성
    if valid_content_ids is not None:
        visit_matrix = visit_matrix[visit_matrix.columns.intersection(valid_content_ids)]

    user_visit_similarity = cosine_similarity(visit_matrix)
    user_visit_similarity_df = pd.DataFrame(user_visit_similarity, index=visit_matrix.index, columns=visit_matrix.index)

    ''' 이 부분에 해인님이 추가해주실 item based 유사도 행렬과 결합 로직 필요 '''

    weight_profile = 0.7 # 설문 데이터 기반 유사도 행렬 가중치
    weight_visit = 0.3 # 방문 기록 기반 유사도 행렬 가중치
    combined_similarity_df = (user_profile_similarity_df * weight_profile + user_visit_similarity_df * weight_visit)

    return df_traveller_encoded, visit_matrix, combined_similarity_df
