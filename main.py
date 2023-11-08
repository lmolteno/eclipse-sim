from astropy.coordinates import EarthLocation, get_body, AltAz, PrecessedGeocentric, ITRS, solar_system_ephemeris
from astropy.utils import iers
import astropy.units as u
from astropy import constants as const
from astropy.time import Time, TimeDelta
import numpy as np
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
import matplotlib.pyplot as plt
from tqdm import tqdm


# observer = EarthLocation.from_geodetic(170.5008807, -45.8792974)
def gcrs_to_earthloc(gcrs_arr, time, lon_offset):
    # gcrs = GCRS(x=gcrs_arr[0] * u.au, y=gcrs_arr[1] * u.au, z=gcrs_arr[2] * u.au, representation_type='cartesian')
    # itrs = gcrs.transform_to(ITRS(obstime=time))
    # return EarthLocation.from_geocentric(itrs.x, itrs.y, itrs.z)
    pre_offset = EarthLocation.from_geocentric(x=gcrs_arr[0], y=gcrs_arr[1], z=gcrs_arr[2], unit=u.km).to_geodetic()
    return EarthLocation.from_geodetic(pre_offset.lon + lon_offset, pre_offset.lat, pre_offset.height)


def get_centerline_pos_at_time(obs_time: Time):
    precessed_frame = ITRS(obstime=obs_time) # PrecessedGeocentric(equinox=obs_time, obstime=obs_time)
    moon_pos = get_body("moon", obs_time)
    moon_pos = moon_pos.transform_to(precessed_frame)
    moon_pos.representation_type = 'cartesian'  # geocentric xyz coords
    moon_pos = np.array([moon_pos.x.to(u.km).value, moon_pos.y.to(u.km).value, moon_pos.z.to(u.km).value])

    sun_pos = get_body("sun", obs_time)
    sun_pos = sun_pos.transform_to(precessed_frame)
    sun_pos.representation_type = 'cartesian'
    sun_pos = np.array([sun_pos.x.to(u.km).value, sun_pos.y.to(u.km).value, sun_pos.z.to(u.km).value])

    # determines direction of line through centers
    dv = (sun_pos - moon_pos) / np.linalg.norm(sun_pos - moon_pos)
    earth_moon_dist = np.linalg.norm(moon_pos)

    starting_point = moon_pos - (dv * earth_moon_dist)
    on_night_side = np.linalg.norm(sun_pos - starting_point) > np.linalg.norm(sun_pos)
    if on_night_side:
        starting_point = starting_point + (dv * const.R_earth.to(u.au).value)

    lon_offset = 0  # (cur_sun_pos.distance / const.c) * (-7.272e-5 * u.rad / u.s)
    centerline_location = gcrs_to_earthloc(starting_point, obs_time, lon_offset)

    iter_limit = 0
    dist = np.abs(centerline_location.height.to(u.m).value)

    while dist > 10 and iter_limit < 100:
        iter_limit += 1
        starting_point = starting_point - (dv * centerline_location.height.to(u.km).value)
        centerline_location = gcrs_to_earthloc(starting_point, obs_time, lon_offset)
        dist = np.abs(centerline_location.height.to(u.m).value)

    if iter_limit == 100:
        return None

    on_night_side = np.linalg.norm(sun_pos - starting_point) > np.linalg.norm(sun_pos)
    return None if on_night_side else centerline_location


solar_system_ephemeris.set('de432s')
iers.conf.iers_auto_url = 'https://datacenter.iers.org/data/9/finals2000A.all'

start_time = Time("2028-07-22 04:18:00")

centerline_points = []
for i in tqdm(range(20)):
    obs_time = start_time - TimeDelta(i * 10.0, format='sec')
    pos = get_centerline_pos_at_time(obs_time)
    local_frame = AltAz(obstime=obs_time, location=pos)
    if pos is not None:
        sun_from_pos = get_body('sun', obs_time).transform_to(local_frame)
        moon_from_pos = get_body('moon', obs_time).transform_to(local_frame)
        local_dist = sun_from_pos.separation(moon_from_pos)
        centerline_points.append(pos)
    else:
        print("No pos")

ax = plt.axes(projection=ccrs.epsg(2193))

x, y = zip(*map(lambda p: [p.lon.value, p.lat.value], centerline_points))
plt.plot(x, y, transform=ccrs.Geodetic())
# ax.coastlines()

nz_coastlines = ShapelyFeature(
    Reader("./coastlines/nz-coastlines-and-islands-polygons-topo-1250k.shp").geometries(),
    ccrs.epsg(2193),
    facecolors=[],
    edgecolors=('black',)
)
ax.add_feature(nz_coastlines)

plt.show()
