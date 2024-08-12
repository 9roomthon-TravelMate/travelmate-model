from sqlalchemy.orm import Session
import models



def get_all_preferences(db: Session, region_id: int = None):
    query = db.query(models.Preference).join(models.Visited, models.Preference.traveler_id == models.Visited.traveler_id)
    if region_id is not None:
        query = query.join(models.TourSpot, models.Visited.content_id == models.TourSpot.content_id)\
                     .filter(models.TourSpot.tour_region_id == region_id)
    return query.all()


def get_all_visited(db: Session, region_id: int = None):
    query = db.query(models.Visited)
    if region_id is not None:
        query = query.join(models.TourSpot, models.Visited.content_id == models.TourSpot.content_id)\
                     .filter(models.TourSpot.tour_region_id == region_id)
    return query.all()
