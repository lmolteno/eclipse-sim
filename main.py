from astropy.coordinates import EarthLocation, get_body, AltAz
import astropy.units as u
from astropy import constants as const
from astropy.time import Time, TimeDelta
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


# observer = EarthLocation.from_geodetic(170.5008807, -45.8792974)
def gcrs_to_earthloc(gcrs_arr, time, lon_offset):
    # gcrs = GCRS(x=gcrs_arr[0] * u.au, y=gcrs_arr[1] * u.au, z=gcrs_arr[2] * u.au, representation_type='cartesian')
    # itrs = gcrs.transform_to(ITRS(obstime=time))
    # return EarthLocation.from_geocentric(itrs.x, itrs.y, itrs.z)
    pre_offset = EarthLocation.from_geocentric(x=gcrs_arr[0], y=gcrs_arr[1], z=gcrs_arr[2], unit=u.au).to_geodetic()
    return EarthLocation.from_geodetic(pre_offset.lon + lon_offset, pre_offset.lat, pre_offset.height)


def get_centerline_pos_at_time(obs_time: Time):
    cur_moon_pos = get_body("moon", obs_time)
    t_moon_offset = (cur_moon_pos.distance / const.c).to(u.s)
    moon_pos = get_body("moon", obs_time + TimeDelta(t_moon_offset.value, format='sec'))
    moon_pos.representation_type = 'cartesian'  # geocentric xyz coords
    moon_pos = np.array([moon_pos.x.value, moon_pos.y.value, moon_pos.z.value])

    cur_sun_pos = get_body("sun", obs_time)
    t_sun_offset = (cur_sun_pos.distance / const.c).to(u.s)
    sun_pos = get_body("sun", obs_time + TimeDelta(t_sun_offset.value, format='sec'))
    sun_pos.representation_type = 'cartesian'
    sun_pos = np.array([sun_pos.x.value, sun_pos.y.value, sun_pos.z.value])

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

    while np.abs(centerline_location.height.to(u.m).value) > 10 and iter_limit < 100:
        iter_limit += 1
        starting_point = starting_point - (dv * centerline_location.height.to(u.au).value)
        centerline_location = gcrs_to_earthloc(starting_point, obs_time, lon_offset)

    if iter_limit == 100:
        return None

    on_night_side = np.linalg.norm(sun_pos - starting_point) > np.linalg.norm(sun_pos)
    return None if on_night_side else centerline_location


start_time = Time("2028-07-22 04:10:38")
centerline_points = []
for i in range(10):
    obs_time = start_time + TimeDelta(i * 60.0, format='sec')
    pos = get_centerline_pos_at_time(obs_time)
    local_frame = AltAz(obstime=obs_time, location=pos)
    if pos is not None:
        sun_from_pos = get_body('sun', obs_time).transform_to(local_frame)
        moon_from_pos = get_body('moon', obs_time).transform_to(local_frame)
        local_dist = sun_from_pos.separation(moon_from_pos)
        print(local_dist.to(u.arcmin).value)
        centerline_points.append(pos)

ax = plt.axes(projection=ccrs.epsg(2193))
ax.coastlines()

lats, lons = zip(*map(lambda p: [p.lat.value, p.lon.value], centerline_points))
plt.plot(lons, lats, transform=ccrs.Geodetic())
plt.show()
