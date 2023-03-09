The US delay dataset is collected from the U.S. Bureau of Transportation Statistics (BTS) database[https://www.transtats.bts.gov/].

udelay.npy --- delay[:,;,0] arrival delay
          --- delay[:,:,1] departure delay

weather2016_2021.npy --- normal weather 0, severe cold 1, fog 2, hail 3, rain 4, snow 5, storm 6, and other precipitation 7

adj_mx.npy --- distance adjacency matrix
od_pair.npy --- origin-destination flow adjacency matrix

USA.csv --- IATA 3-Letter Codes of 70 USA Airports. The order is corresponding their order in the npys. 
