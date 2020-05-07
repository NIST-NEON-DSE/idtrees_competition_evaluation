import geopandas
import rasterio

# this will go in a loop for each site and each test plot. Now hardcoding
site = "OSBS"
plot_n = '24'
detection_path = './submission/'+site+'_submission.csv'
ras_path = './RS/'+site+'_'+plot_n+'.tif'
raster = rasterio.open(ras_path)

#if using RGB, we have 10 pixels per meter: if we are evaluating on RGB, we probably need to take into account if using w/l
pix_per_meter = 10
#automatically load only boxes within plots boundaries
gdf = geopandas.read_file(
    detection_path,
    bbox=raster.bounds,
)
gtf = geopandas.read_file(
    './submission/'+site+'_ground.csv',
    bbox=raster.bounds,
)

# turn WTK into coordinates within in the image
gdf_limits = gdf.bounds
gtf_limits = gtf.bounds

xmin = raster.bounds[0]
ymin = raster.bounds[1]

#length
gdf_limits['maxy'] = (gdf_limits['maxy'] - gdf_limits['miny'])*pix_per_meter
gtf_limits['maxy'] = (gtf_limits['maxy'] - gtf_limits['miny'])*pix_per_meter

#width
gdf_limits['maxx'] = (gdf_limits['maxx'] - gdf_limits['minx'])*pix_per_meter
gtf_limits['maxx'] = (gtf_limits['maxx'] - gtf_limits['minx'])*pix_per_meter

# translate coords to 0,0
gdf_limits['minx'] = (gdf_limits['minx'] - xmin) * pix_per_meter
gdf_limits['miny'] = (gdf_limits['miny'] - ymin) * pix_per_meter
gdf_limits.columns = ['minx', 'miny', 'width', 'length']

#same for groundtruth
gtf_limits['minx'] = (gtf_limits['minx'] - xmin) * pix_per_meter
gtf_limits['miny'] = (gtf_limits['miny'] - ymin) * pix_per_meter
gtf_limits.columns = ['minx', 'miny', 'width', 'length']

gdf_limits[gdf_limits < 0] = 0
gtf_limits[gtf_limits < 0] = 0

gdf_limits = gdf_limits.astype(int)
gtf_limits = gtf_limits.astype(int)

#save edges as a new csv file to be fed into the evaluation code
gdf_limits.to_csv('./eval/'+site+'_'+plot_n+'_detections_edges.csv')
gtf_limits.to_csv('./eval/'+site+'_'+plot_n+'_groundtruth_edges.csv')






