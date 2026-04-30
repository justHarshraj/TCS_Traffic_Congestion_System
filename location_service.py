import requests

def get_device_location():
    try:
        response = requests.get("https://ipinfo.io/json", timeout=5)
        data = response.json()

        coords = data.get("loc", "")  # format: "lat,long"

        if coords:
            latitude, longitude = coords.split(",")
        else:
            latitude, longitude = "Unknown", "Unknown"

        location_text = f"{latitude}, {longitude}"
        map_link = f"https://maps.google.com/?q={latitude},{longitude}"

        return latitude, longitude, map_link

    except Exception as e:
        print("Location detection failed:", e)
        return "Unknown", "Unknown", "Unavailable"
