import pandas as pd
from sncast import minML
import pygmt

if __name__ == '__main__':

    noise = pd.read_csv('/Users/eart0593/Projects/NEP_consulting/notebooks/uk_core_network_P95_noise_est_2023_2024.csv')

    data = minML(noise, stat_num=5, foc_depth=5, lon0=-3,
                                           lon1=3, lat0=53, lat1=56, dlat=1, dlon=1,
                                           region='UK') 
    

    region = [-2,2,53,56]
    # pygmt.xyz2grd(data=df, outgrid='test.grd', spacing=(0.025,0.025), region=region)
    figure = pygmt.Figure()
    figure.basemap(projection='M12c',region=region,
                    frame=["af"])
    
    # figure.coast(shorelines=True, water='skyblue', borders='1',rivers=2, map_scale='g-1./53.3+w50k+c2/54+f+l')
    figure.grdcontour(grid=data[:].T, interval=0.25, annotation=0.5)
    figure.grdimage(data[:].T)

    figure.colorbar(frame=['x+lM@-L@-'])
    figure.show()
