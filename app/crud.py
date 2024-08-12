from sqlalchemy.orm import Session
import models

def get_all_preferences(db: Session):
    return db.query(models.Preference).all()

def get_all_visited(db: Session):
    return db.query(models.Visited).all()

def get_tour_spots(db: Session, region_id: int = None):
    query = db.query(models.TourSpot)
    if region_id is not None:
        query = query.filter(models.TourSpot.tour_region_id == region_id)
    return query.all()
