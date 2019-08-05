## Performance Evaluation 1
TTC running result:

| lidar based | Cemera based |
|------------|--------------|
| 12\.5156   | 13\.6884     |
| 12\.5141   | 13\.0114     |
| 14\.2179   | 11\.4703     |
| 16\.6894   | 12\.4201     |
| 15\.9082   | 11\.6524     |
| 12\.5748   | **24\.1508**     |
| 11\.9836   | 11\.7588     |
| 13\.2382   | 13\.2405     |
| 13\.0241   | 11\.4727     |
| 11\.1746   | 13\.9253     |
| 12\.8086   | 10\.8266     |
| 8\.95978   | 11\.862      |
| 9\.96439   | 11\.376      |
| 9\.59863   | 12\.0086     |
| 8\.57352   | 9\.65331     |
| 9\.51617   | 10\.9372     |
| 9\.54658   | 11\.2178     |
| 8\.3988    | 8\.75986     |

![no txt](images/TTC_result.png)

As the gloabl trend is as we expected descending order.

The Lidar based reulst is not as bumper as Cemera based. As you could see there is still outliers in the bBox:

![no txt](images/outlier_01.png)

![no txt](images/outlier_02.png)

We just simply use the median point instead of mean which it is more affected by the outlier.

And formore, we could add more lidar points as the Lidar clustring(RANSAC) algorithm could be used first to clean and cluster the lidar data and then combined with bBox to compute TTC.

The cemera based result is more bumper as the process may contain some mismatched points as those points will be included in computing the results.

We may use varies of image feature detector/extractor pair to check the optimal result.


## Performance Evaluation 2

