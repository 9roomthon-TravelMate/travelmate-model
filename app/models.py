# from sqlalchemy import Column, Integer, BigInteger, String, Date
# from database import Base
from sqlalchemy import Column, Integer, BigInteger, String, Date
from .database import Base

# 사용자 취향 테이블 참조
class Preference(Base):
    __tablename__ = 'preference'

    preference_id = Column(Integer, primary_key=True, index=True)
    traveler_id = Column(BigInteger, index=True)
    gender = Column(String(10), nullable=False)
    age_grp = Column(Integer, nullable=False)
    travel_start_ymd = Column(Date, nullable=False)
    travel_end_ymd = Column(Date, nullable=False)
    travel_styl_1 = Column(Integer, nullable=False)
    travel_styl_2 = Column(Integer, nullable=False)
    travel_styl_3 = Column(Integer, nullable=False)
    travel_styl_4 = Column(Integer, nullable=False)
    travel_styl_5 = Column(Integer, nullable=False)
    travel_styl_6 = Column(Integer, nullable=False)
    travel_styl_7 = Column(Integer, nullable=False)
    travel_companions_num = Column(Integer, nullable=False)

# 방문지 목록 테이블 참조
class Visited(Base):
    __tablename__ = 'visited'

    visit_id = Column(Integer, primary_key=True, index=True)
    traveler_id = Column(BigInteger, nullable=False)
    content_id = Column(String(255), nullable=False)

# 지역 필터링을 위해 tour_spot 테이블도 참조
class TourSpot(Base):
    __tablename__ = 'tour_spot'

    content_id = Column(String(255), primary_key=True, index=True)
    tour_region_id = Column(Integer, nullable=False)