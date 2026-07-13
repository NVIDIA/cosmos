import functools
import logging
import math
import random

from geopy.distance import distance
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

USER_AGENT_SUFFIX = hex(random.getrandbits(128))[2:]
geolocator = Nominatim(user_agent=f"vlm-mega-benchmark_{USER_AGENT_SUFFIX}")


error_logger = logging.getLogger("errorLogger")


def calculate_proximity_score(guess_coords, actual_coords, k=100):
    """Calculate the proximity score based on the location.

    Exponentially decreases depending on the distance.

    Args:
        guess_coords (float, float): The longitude and latitude of the guessed coordinates.
        actual_coords (float, float): The longitude and latitude of the actual coordinates.
        k (numbers.Number): The threshold (in km) at which we get a score of 0.5.
    """
    dist = distance(guess_coords, actual_coords).km
    proximity_score = math.exp(-dist / k)
    return proximity_score


GEOLOCATION_TIMEOUT = 1
MAX_RETRIES = 30


geocode = RateLimiter(
    geolocator.geocode, min_delay_seconds=GEOLOCATION_TIMEOUT, max_retries=MAX_RETRIES
)


@functools.cache
def try_geolocate(query):
    """Try to look up the location."""
    location = geocode(query)
    if location is None:
        error_logger.error(
            f"Geolocation API request failed due to timeout: exceeded {MAX_RETRIES} retries!"
        )
    return location


def location_to_coords(
    country: str, province_or_state: str, municipality: str
) -> tuple[float, float] | None:
    if country == "" or province_or_state == "" or municipality == "":
        return None
    """Convert the location to longitude and latitude."""
    location = geolocator.geocode(
        query={"country": country, "state": province_or_state, "city": municipality}
    )
    if location is not None:
        return (location.latitude, location.longitude)
    # Try searching without the province/state, as it can be non-standard for some questions
    location = geolocator.geocode(query={"country": country, "city": municipality})
    if location is None:
        return None
    return (location.latitude, location.longitude)


class GeoProximityLocationDict:
    """Return a score based on the distance between two locations."""

    @classmethod
    def match(cls, responses, targets) -> float:
        """Return a score based on how far two targets are away from each other,
        where each field is a dict with the following schema:
        {
            country: str,
            province_or_state: str,
            municipality: str
        }
        """
        try:
            guess_coords = location_to_coords(**responses)
        except:
            return 0

        if guess_coords is None:
            error_logger.error(
                f"GeoProximityLocationDict: could not load co-ordinates for {responses=}"
            )
            return 0
        actual_coords = location_to_coords(**targets)
        if actual_coords is None:
            error_logger.error(
                f"GeoProximityLocationDict: could not load co-ordinates for {targets=}"
            )
            return 0

        return calculate_proximity_score(guess_coords, actual_coords)
