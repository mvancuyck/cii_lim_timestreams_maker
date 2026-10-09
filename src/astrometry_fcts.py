import numpy as np
import matplotlib.pyplot as plt 
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from IPython import embed

#--- Dan Functions ---- 

def zenithAngle(dec,lat,HA):
    """
    source zenith angle (rad)

    Parameters
    ----------
    dec: float 
        declination angle in degrees     
    lat: float
        latitude   angle in degrees
    HA: array
        hour angle in hour

    Returns
    -------
    za: array
        zenith angle in degree
    """ 
    za = np.arccos(np.sin(np.radians(lat)) * np.sin(np.radians(dec)) + np.cos(np.radians(lat)) * np.cos(np.radians(dec)) * np.cos(HA* np.pi/12))
    return za

def elevationAngle(dec,lat,HA): 
    """
    elevation angle (rad)

    Parameters
    ----------
    dec: float 
        declination angle in degrees     
    lat: float
        latitude angle in degrees
    HA: array
        hour angle in hour

    Returns
    -------
    ea: array
        elevation angle in degree
    """ 

    return np.pi/2 - zenithAngle(dec,lat,HA)

def azimuthAngle(dec,lat,HA):
    """
    source azimuth angle (rad)

    Parameters
    ----------
    dec: float 
        declination angle in degrees     
    lat: float
        latitude angle in degrees
    HA: array
        hour angle in hour

    Returns
    -------
    aa: array
        source azimuth angle (rad)
    """ 

    za = zenithAngle(dec,lat,HA)
    cosAz = (np.sin(np.radians(dec)) - np.sin(np.radians(lat)) * np.cos(za))/(np.cos(np.radians(lat)) * np.sin(za))
    sinAz = - np.sin(HA * np.pi/12) * np.cos(np.radians(dec)) / np.sin(za)
    return np.arctan2(sinAz,cosAz)

def parallacticAngle(dec,lat,HA,unwrapPA=True):
    """
    source parrallactic angle (rad)

    Parameters
    ----------
    dec: float 
        declination angle in degrees     
    lat: float
        latitude angle in degrees
    HA: array
        hour angle in degree

    Returns
    -------
    pa: array
        source parrallactic angle (rad)
    """ 
    pa = np.arctan2(np.sin(HA * np.pi/12), np.cos(np.radians(dec)) * np.tan(np.radians(lat)) - np.sin(np.radians(dec)) * np.cos(HA * np.pi/12))
    if unwrapPA == True:
        pa = np.unwrap(pa)
        if np.mean(pa)<0:
            pa = pa + 2*np.pi
    return pa

def declinationAngle(azi, alt, lat):
    """
    source declination angle (rad)

    Parameters
    ----------
    azi: float 
        azimuth in degrees     
    alt: float
        latitude  angle in degrees
    lat: float
        latitude angle in degree

    Returns
    -------
    Dec: float
        source declination angle (rad)
    """ 
    sinDec =  np.sin(np.radians(alt))*np.sin(np.radians(lat)) + np.cos(np.radians(alt))*np.cos(np.radians(lat))*np.cos(np.radians(azi))
    return np.arcsin(sinDec)

def hourAngle(azi, alt, lat):
    """
    source hour angle (rad)

    Parameters
    ----------
    azi: float 
        azimuth in degrees     
    alt: float
        latitude angle in degrees
    lat: float
        latitude angle in degree

    Returns
    -------
    ha: float
        source hour angle (rad)
    """ 
    
    tanHA = - np.sin(np.radians(azi)) / (np.tan(np.radians(alt)) * np.cos(np.radians(lat)) - np.cos(np.radians(azi))*np.sin(np.radians(lat)))
    HA = np.arctan(tanHA)

    '''
    sin_dec = np.sin(np.radians(alt))*np.sin(np.radians(lat)) + np.cos(np.radians(alt))*np.cos(np.radians(lat))*np.cos(np.radians(azi))
    dec = np.arcsin(sin_dec)
    
    sin_HA = -np.sin(np.radians(azi))*np.cos(np.radians(alt)) / np.cos(dec)
    cos_HA = (np.sin(np.radians(alt)) - np.sin(dec)*np.sin(np.radians(lat))) / (np.cos(dec)*np.cos(np.radians(lat)))
    HA = np.arctan2(sin_HA, cos_HA)
    '''

    return HA


if __name__ == "__main__":

    # HA = np.arange(-5,5,0.04)
    # lat=np.radians(19.82547)
    # dec=np.radians(16.14821)
    # pa = parallacticAngle(dec,lat,HA)
    # el = elevationAngle(dec,lat,HA)
    # za = zenithAngle(dec,lat,HA)

    mcmLat = -77.83
    minLat = -85
    maxLat = -75
    latList = np.arange(minLat,maxLat,2.)
    from astropy.coordinates import SkyCoord

    # http://simbad.u-strasbg.fr/simbad/sim-id?Ident=GOODS+South+field
    c=SkyCoord.from_name('Goods-S Field')
    print('GOODS-S:   {0:}'.format(c.to_string('hmsdms')))
    goodsSDec = c.dec.value
    sptDeepDec = -55.

    HA = np.arange(-12,12,.02)

    elMCMgoods = elevationAngle(goodsSDec,mcmLat,HA)
    paMCMgoods = parallacticAngle(goodsSDec,mcmLat,HA)
    elMCMspt = elevationAngle(sptDeepDec,mcmLat,HA)
    paMCMspt = parallacticAngle(sptDeepDec,mcmLat,HA)


    elMin = np.radians(30.)
    plt.clf()
    plt.plot(HA[elMCMgoods>elMin],np.degrees(paMCMgoods[elMCMgoods>elMin]),label='GOODS-S')
    plt.plot(HA[elMCMspt>elMin],np.degrees(paMCMspt[elMCMspt>elMin]),label='SPTDeep')
    plt.xlabel('Hour Angle (h)'); #plt.xticks(np.arange(-6,6.1,1));
    plt.ylabel('Parallactic Angle (deg)')
    plt.legend(handlelength=1,loc='best',fontsize='small')
    plt.title('PA from MCM')
    plt.savefig('figs/mcmPA')

    plt.clf()
    for latVal in latList:
        elVals = elevationAngle(goodsSDec,latVal,HA)
        paVals = parallacticAngle(goodsSDec,latVal,HA)
        plt.plot(HA[elVals>elMin],np.degrees(paVals[elVals>elMin]),label=str(latVal))
    plt.xlabel('Hour Angle (h)'); plt.xticks(np.arange(-6,6.1,1));
    plt.ylabel('Parallactic Angle (deg)')
    plt.legend(handlelength=1,loc='best',fontsize='small',title='Latitude')
    plt.title('GOODS-S PA')
    plt.savefig('figs/goodsPA')

    plt.clf()
    for latVal in latList:
        elVals = elevationAngle(sptDeepDec,latVal,HA)
        paVals = parallacticAngle(sptDeepDec,latVal,HA)
        plt.plot(HA[elVals>elMin],np.degrees(paVals[elVals>elMin]),label=str(latVal))
    plt.xlabel('Hour Angle (h)'); # plt.xticks(np.arange(-12,12.1,1));
    plt.ylabel('Parallactic Angle (deg)')
    plt.legend(handlelength=1,loc='best',fontsize='small',title='Latitude')
    plt.title('SPT-Deep PA')
    plt.savefig('figs/sptPA')
	
    plt.show()