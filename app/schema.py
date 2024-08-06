from pydantic import BaseModel
from datetime import date

class PreferenceBase(BaseModel):
    gender: str
    age_grp: int
    travel_start_ymd: date
    travel_end_ymd: date
    travel_styl_1: int
    travel_styl_2: int
    travel_styl_3: int
    travel_styl_4: int
    travel_styl_5: int
    travel_styl_6: int
    travel_styl_7: int
    travel_companions_num: int

class PreferenceCreate(PreferenceBase):
    traveler_id: int

class Preference(PreferenceBase):
    preference_id: int
    traveler_id: int

    class Config:
        orm_mode = True

class VisitedBase(BaseModel):
    content_id: str

class VisitedCreate(VisitedBase):
    traveler_id: int

class Visited(VisitedBase):
    visit_id: int
    traveler_id: int

    class Config:
        orm_mode = True
