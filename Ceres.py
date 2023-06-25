
# Uvoz paketa
import spiceypy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt





# Uvoz auxiliary pod-modula koji sadrži većinu postavljenih formula
import sys
sys.path.insert(1, '../_auxiliary')
import asteroid_aux

# Ceres NAIF ID
CERES_ID = 2000001

# Učitavanje jezgra
spiceypy.furnsh('kernel_meta.txt')

# Geografska širina i dužina dobijena iz baze https://ssd.jpl.nasa.gov/sbdb.cgi#top
CERES_ABS_MAG = 3.4
CERES_SLOPE_G = 0.12

# Postavljanje pandas formata
ceres_df = pd.DataFrame([])

#Učitavanje jezgra i potrebnih dodatnih podataka
spiceypy.furnsh('naif0012.tls')
spiceypy.furnsh('de432s.bsp')
spiceypy.furnsh('codes_300ast_20100725.bsp')
spiceypy.furnsh('codes_300ast_20100725.cmt')
spiceypy.furnsh('codes_300ast_20100725.tf')



# Postavljanje vremenskog formata
DATETIME_RANGE = pd.date_range(start='2013-01-01T00:00:00', \
                               end='2013-12-31T00:00:00', \
                               freq='1W')


ceres_df.loc[:, 'UTC_TIME'] = DATETIME_RANGE


ceres_df.loc[:, 'UTC_PARSED'] = DATETIME_RANGE.strftime('%Y-%j')


ceres_df.loc[:, 'ET_TIME'] = ceres_df['UTC_TIME'] \
                                 .apply(lambda x: spiceypy.utc2et(str(x)))

# Računanje razdaljine između Ceresa i Sunca, izraženu u km
ceres_df.loc[:, 'DIST_SUN_AU'] = \
    ceres_df['ET_TIME'].apply(lambda x: \
        spiceypy.convrt( \
            spiceypy.vnorm( \
                spiceypy.spkgps(targ=CERES_ID, \
                                et=x, \
                                ref='ECLIPJ2000', \
                                obs=10)[0]), \
            'km', 'AU'))

# Računanje razdaljine između Ceresa i Zemlje izraženu u km
ceres_df.loc[:, 'DIST_EARTH_AU'] = \
    ceres_df['ET_TIME'].apply(lambda x: \
        spiceypy.convrt( \
            spiceypy.vnorm( \
                spiceypy.spkgps(targ=CERES_ID, \
                                et=x, \
                                ref='ECLIPJ2000', \
                                obs=399)[0]), \
            'km', 'AU'))

# Računanje ugla 
ceres_df.loc[:, 'PHASE_ANGLE_EARTH2SUN_RAD'] = \
    ceres_df['ET_TIME'].apply(lambda x: spiceypy.phaseq(et=x, \
                                                        target=str(CERES_ID), \
                                                        illmn='10', \
                                                        obsrvr='399', \
                                                        abcorr='NONE'))


ceres_df.loc[:, 'PHASE_ANGLE_EARTH2SUN_DEG'] = \
    np.degrees(ceres_df['PHASE_ANGLE_EARTH2SUN_RAD'])


# Računanje prividne veličine
ceres_df.loc[:, 'APP_MAG'] = \
    ceres_df.apply(lambda x: \
        asteroid_aux.app_mag(abs_mag=CERES_ABS_MAG, \
                             phase_angle=x['PHASE_ANGLE_EARTH2SUN_RAD'], \
                             slope_g=CERES_SLOPE_G, \
                             d_ast_sun=x['DIST_SUN_AU'], \
                             d_ast_earth=x['DIST_EARTH_AU']), \
        axis=1)

# Postavljanje eliptičkih koordinata
ceres_df.loc[:, 'ECLIP_LONG_RAD'] = \
    ceres_df['ET_TIME'].apply(lambda x: \
        spiceypy.recrad(spiceypy.spkgps(targ=CERES_ID, \
                                        et=x, \
                                        ref='ECLIPJ2000', \
                                        obs=399)[0])[1])


ceres_df.loc[:, 'ECLIP_LAT_RAD'] = \
    ceres_df['ET_TIME'].apply(lambda x: \
        spiceypy.recrad(spiceypy.spkgps(targ=CERES_ID, \
                                        et=x, \
                                        ref='ECLIPJ2000', \
                                        obs=399)[0])[2])

# Konvertovanje u stepene
ceres_df.loc[:, 'ECLIP_LONG_DEG'] = \
    np.degrees(ceres_df['ECLIP_LONG_RAD'])

ceres_df.loc[:, 'ECLIP_LAT_DEG'] = \
    np.degrees(ceres_df['ECLIP_LAT_RAD'])

# Slikoviti prikaz
from sklearn import preprocessing


PRE_SCALED = np.array(np.min(ceres_df['APP_MAG']) \
                      / ceres_df['APP_MAG'].values).reshape(-1, 1)


scaler = preprocessing.MinMaxScaler()


scaler.fit(PRE_SCALED)


marker_pre_scale = scaler.transform(PRE_SCALED)


marker_scale = marker_pre_scale * 50
ceres_df.loc[:, 'PLOT_MARKER_SIZE'] = marker_scale


plt.style.use('dark_background')


plt.figure(figsize=(12, 8))


cm = plt.cm.get_cmap('viridis_r')


plt.scatter(x=ceres_df['ECLIP_LONG_DEG'], \
            y=ceres_df['ECLIP_LAT_DEG'], \
            c=ceres_df['APP_MAG'], \
            alpha=1, \
            s=ceres_df['PLOT_MARKER_SIZE'], \
            marker='o', \
            cmap=cm)

# Bijela linija
plt.plot(ceres_df['ECLIP_LONG_DEG'], \
         ceres_df['ECLIP_LAT_DEG'], \
         marker=None, \
         linestyle='dashed', \
         color='white', \
         alpha=0.3, \
         lw=1)


for date_str, ceres_x, ceres_y in ceres_df[['UTC_PARSED', \
                                            'ECLIP_LONG_DEG', \
                                            'ECLIP_LAT_DEG']].values[2::10]:

    
    plt.annotate(date_str,
                 (ceres_x, ceres_y),
                 textcoords="offset points",
                 xytext=(12, 2),
                 ha='left',
                 color='white', \
                 alpha=0.7, \
                 fontsize=8)

plt.grid(True, linestyle='dashed', alpha=0.5)


ax = plt.gca()
ax.ticklabel_format(useOffset=False, style='plain')


cbar = plt.colorbar()
cbar.ax.invert_yaxis()
cbar.set_label('Prividna velicina')


plt.xlabel('Elipticka duzina')
plt.ylabel('Elipticka sirina')


plt.xlim(np.min(ceres_df['ECLIP_LONG_DEG'])*0.98, \
         np.max(ceres_df['ECLIP_LONG_DEG'])*1.02)


plt.title('Kretanje planete Ceres na nebu')


plt.savefig('ceres_sky_map_movement2023.png', dpi=300)