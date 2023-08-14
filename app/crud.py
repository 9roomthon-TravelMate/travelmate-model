from sqlalchemy.orm import Session
from app import models


def get_all_preferences(db: Session, traveler_id: int =None, region_id: int = None):
    
    query = db.query(models.Preference).join(models.Visited, models.Preference.traveler_id == models.Visited.traveler_id)
    result = set()

    if traveler_id is not None:
        traveler_query = db.query(models.Preference).filter(models.Preference.traveler_id == traveler_id)
        traveler_results = traveler_query.all()
        print(f"Traveler Results for ID {traveler_id}: {traveler_results}")  
        result.update(traveler_results)
    
    if region_id is not None:
        region_query = query.join(models.TourSpot, models.Visited.content_id == models.TourSpot.content_id)\
                            .filter(models.TourSpot.tour_region_id == region_id)
        region_results = region_query.all()
        #print(f"Region Results for ID {region_id}: {region_results}")  
        result.update(region_results)
    
    return list(result)


def get_all_visited(db: Session, region_id: int = None):
    query = db.query(models.Visited)
    if region_id is not None:
        query = query.join(models.TourSpot, models.Visited.content_id == models.TourSpot.content_id)\
                     .filter(models.TourSpot.tour_region_id == region_id)
    return query.all()
