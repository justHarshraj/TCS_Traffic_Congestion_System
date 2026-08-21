def get_device_location():
    """Return the fixed device location coordinates."""
    latitude = "23.064254212172607"
    longitude = "72.43983468434534"
    map_link = f"https://maps.google.com/?q={latitude},{longitude}"

    return latitude, longitude, map_link
