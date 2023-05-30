#### https://www.kaggle.com/marcuswingen/eda-of-bookings-and-ml-to-predict-cancelations

## Preparing our Environment 


```python
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

```


```python
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px
import cufflinks as cf
cf.go_offline()
```


<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-2.9.0.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>




```python
!pip install sort-dataframeby-monthorweek
 
#Dependency package needs to be installed
!pip install sorted-months-weekdays
```

    Requirement already satisfied: sort-dataframeby-monthorweek in c:\users\ro2ya\anaconda3\lib\site-packages (0.4)
    Requirement already satisfied: sorted-months-weekdays in c:\users\ro2ya\anaconda3\lib\site-packages (0.2)
    

#### Read Hotel Bookings data


```python
df=pd.read_csv('hotel_bookings.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>arrival_date_week_number</th>
      <th>arrival_date_day_of_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>...</th>
      <th>deposit_type</th>
      <th>agent</th>
      <th>company</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>reservation_status</th>
      <th>reservation_status_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>342</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/1/2015</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>737</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/1/2015</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>7</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/2/2015</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>13</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>304.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/2/2015</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>14</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>240.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.0</td>
      <td>0</td>
      <td>1</td>
      <td>Check-Out</td>
      <td>7/3/2015</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>





### Understanding our data


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 119390 entries, 0 to 119389
    Data columns (total 32 columns):
     #   Column                          Non-Null Count   Dtype  
    ---  ------                          --------------   -----  
     0   hotel                           119390 non-null  object 
     1   is_canceled                     119390 non-null  int64  
     2   lead_time                       119390 non-null  int64  
     3   arrival_date_year               119390 non-null  int64  
     4   arrival_date_month              119390 non-null  object 
     5   arrival_date_week_number        119390 non-null  int64  
     6   arrival_date_day_of_month       119390 non-null  int64  
     7   stays_in_weekend_nights         119390 non-null  int64  
     8   stays_in_week_nights            119390 non-null  int64  
     9   adults                          119390 non-null  int64  
     10  children                        119386 non-null  float64
     11  babies                          119390 non-null  int64  
     12  meal                            119390 non-null  object 
     13  country                         118902 non-null  object 
     14  market_segment                  119390 non-null  object 
     15  distribution_channel            119390 non-null  object 
     16  is_repeated_guest               119390 non-null  int64  
     17  previous_cancellations          119390 non-null  int64  
     18  previous_bookings_not_canceled  119390 non-null  int64  
     19  reserved_room_type              119390 non-null  object 
     20  assigned_room_type              119390 non-null  object 
     21  booking_changes                 119390 non-null  int64  
     22  deposit_type                    119390 non-null  object 
     23  agent                           103050 non-null  float64
     24  company                         6797 non-null    float64
     25  days_in_waiting_list            119390 non-null  int64  
     26  customer_type                   119390 non-null  object 
     27  adr                             119390 non-null  float64
     28  required_car_parking_spaces     119390 non-null  int64  
     29  total_of_special_requests       119390 non-null  int64  
     30  reservation_status              119390 non-null  object 
     31  reservation_status_date         119390 non-null  object 
    dtypes: float64(4), int64(16), object(12)
    memory usage: 29.1+ MB
    


```python
# Checking if there are null values
df.isnull().sum()
```




    hotel                                  0
    is_canceled                            0
    lead_time                              0
    arrival_date_year                      0
    arrival_date_month                     0
    arrival_date_week_number               0
    arrival_date_day_of_month              0
    stays_in_weekend_nights                0
    stays_in_week_nights                   0
    adults                                 0
    children                               4
    babies                                 0
    meal                                   0
    country                              488
    market_segment                         0
    distribution_channel                   0
    is_repeated_guest                      0
    previous_cancellations                 0
    previous_bookings_not_canceled         0
    reserved_room_type                     0
    assigned_room_type                     0
    booking_changes                        0
    deposit_type                           0
    agent                              16340
    company                           112593
    days_in_waiting_list                   0
    customer_type                          0
    adr                                    0
    required_car_parking_spaces            0
    total_of_special_requests              0
    reservation_status                     0
    reservation_status_date                0
    dtype: int64



**Dealing with null values**


```python
df.fillna(0, inplace=True)
```


```python

df['meal'].value_counts()
```




    BB           92310
    HB           14463
    SC           10650
    Undefined     1169
    FB             798
    Name: meal, dtype: int64




- BB: Bed & Breakfast
- HB: Half Board (Breakfast and Dinner normally)
- FB: Full Board (Beakfast, Lunch and Dinner)
- SC & Undefined : no meal


```python
df['children'].unique()
```




    array([ 0.,  1.,  2., 10.,  3.])




```python
df['adults'].unique()
```




    array([ 2,  1,  3,  4, 40, 26, 50, 27, 55,  0, 20,  6,  5, 10],
          dtype=int64)




```python
df['babies'].unique()
```




    array([ 0,  1,  2, 10,  9], dtype=int64)



**Removing bookings with zero persons**


```python
filter = (df['children']==0) & (df['adults']==0) & (df['babies']==0)
data = df[~filter]  #~ used to get the oppisite of the filter
```


```python
resort = data[(data['hotel'] == 'Resort Hotel') & (data['is_canceled'] == 0)]
city = data[(data['hotel'] == 'City Hotel') & (data['is_canceled'] == 0)]
```

**Analyzing The countries of Resort Hotel** <br>
Where the visitors come from?


```python
labels = resort['country'].value_counts().index
values = resort['country'].value_counts()
```


```python
import plotly.graph_objs as go
from plotly.offline import iplot
import plotly.express as px
```


```python
trace=go.Pie(labels=labels, values=values,
               hoverinfo='label+percent', textinfo='value+label'
               )
```


```python
iplot([trace])
```


<div>                            <div id="ef5fea67-56a9-4dc5-bcea-beedc903d4ef" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("ef5fea67-56a9-4dc5-bcea-beedc903d4ef")) {                    Plotly.newPlot(                        "ef5fea67-56a9-4dc5-bcea-beedc903d4ef",                        [{"hoverinfo":"label+percent","labels":["PRT","GBR","ESP","IRL","FRA","DEU","CN","NLD",0,"USA","BEL","ITA","BRA","CHE","POL","SWE","AUT","ROU","FIN","RUS","CHN","NOR","AUS","DNK","LUX","ARG","LTU","MAR","HUN","IND","LVA","EST","UKR","CZE","AGO","ISR","TUR","NZL","CHL","ZAF","COL","TWN","OMN","SVN","MYS","HRV","JPN","SVK","PRI","NGA","GRC","PHL","SRB","GIB","DZA","KOR","MEX","ISL","THA","CYP","BGR","URY","IRN","JAM","CPV","GEO","CUB","KAZ","SGP","BLR","SUR","MOZ","ARE","LBN","IDN","CAF","DOM","MDV","PAK","AND","KWT","CRI","MLT","ZWE","VEN","JOR","AZE","CIV","ECU","ARM","MWI","ALB","CMR","VNM","MDG","BWA","LKA","UZB","NPL","MAC","TGO","HKG","DJI","BHS","PLW","PER","EGY","QAT","ZMB","MKD","SMR","BDI","SYR","CYM","UGA","COM","MUS","BIH","SAU"],"textinfo":"value+label","values":[10184,5922,3105,1734,1399,1057,614,458,419,407,389,379,329,323,294,231,176,145,135,128,125,100,72,60,54,48,45,39,38,33,29,29,23,23,17,16,16,14,14,13,12,12,11,10,10,9,9,9,9,8,8,7,7,7,7,6,6,6,6,6,5,5,5,5,5,4,4,4,4,4,4,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],"type":"pie"}],                        {"template":{"data":{"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('ef5fea67-56a9-4dc5-bcea-beedc903d4ef');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
country_wise_data=data[data['is_canceled']==0]['country'].value_counts().reset_index()

```


```python
country_wise_data.columns=['Country','No of guests']
country_wise_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>No of guests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PRT</td>
      <td>20977</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GBR</td>
      <td>9668</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FRA</td>
      <td>8468</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ESP</td>
      <td>6383</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DEU</td>
      <td>6067</td>
    </tr>
  </tbody>
</table>
</div>




```python
px.choropleth(country_wise_data,
             locations=country_wise_data['Country'],
             color=country_wise_data['No of guests'],
           #  hover_name=country_wise_data['Country'],
            title='Home Country for Guests'
             )
```


<div>                            <div id="e391ca34-2231-476d-b0e8-7e0591531fda" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("e391ca34-2231-476d-b0e8-7e0591531fda")) {                    Plotly.newPlot(                        "e391ca34-2231-476d-b0e8-7e0591531fda",                        [{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Country=%{location}<br>No of guests=%{z}<extra></extra>","locations":["PRT","GBR","FRA","ESP","DEU","IRL","ITA","BEL","NLD","USA","BRA","CHE","AUT","CN","SWE","POL","CHN","ISR","NOR",0,"RUS","FIN","ROU","DNK","AUS","LUX","JPN","ARG","AGO","HUN","MAR","TUR","CZE","IND","SRB","GRC","DZA","KOR","MEX","HRV","LTU","NZL","EST","BGR","IRN","ISL","ZAF","CHL","COL","UKR","MOZ","LVA","SVN","SVK","THA","CYP","TWN","MYS","URY","PER","SGP","LBN","EGY","TUN","ECU","JOR","CRI","BLR","PHL","SAU","OMN","IRQ","VEN","KAZ","NGA","MLT","CPV","IDN","CMR","BIH","PRI","ALB","KWT","BOL","PAN","ARE","GNB","MKD","LBY","CUB","AZE","GEO","GIB","LKA","VNM","MUS","ARM","JAM","DOM","CAF","SUR","PAK","GTM","KEN","BRB","CIV","PRY","QAT","SYR","MCO","SEN","HKG","BGD","MNE","MDV","ABW","RWA","SLV","TZA","GAB","TMP","GHA","ATA","LIE","LAO","MWI","ETH","TGO","ZWE","COM","AND","UZB","UGA","STP","KNA","MAC","MRT","BWA","SMR","ZMB","ASM","NCL","GUY","KIR","SDN","ATF","TJK","SLE","CYM","LCA","PYF","BHS","DMA","MMR","AIA","BDI","BFA","PLW","SYC","MDG","NAM","BHR","DJI","MLI","NPL","FRO"],"name":"","z":[20977,9668,8468,6383,6067,2542,2428,1868,1716,1592,1392,1298,1033,1025,793,703,537,500,426,421,391,377,366,326,319,177,169,160,157,153,150,146,134,116,98,93,82,78,75,75,74,68,65,63,59,53,49,49,48,48,48,46,41,41,41,40,37,25,23,23,22,22,21,20,19,18,18,17,15,15,14,14,14,14,13,13,12,11,10,10,10,10,10,10,9,8,8,8,8,8,8,7,7,7,6,6,6,6,6,5,5,5,4,4,4,4,4,4,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],"type":"choropleth"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"geo":{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"center":{}},"coloraxis":{"colorbar":{"title":{"text":"No of guests"}},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"legend":{"tracegroupgap":0},"title":{"text":"Home Country for Guests"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('e391ca34-2231-476d-b0e8-7e0591531fda');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>arrival_date_week_number</th>
      <th>arrival_date_day_of_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>...</th>
      <th>deposit_type</th>
      <th>agent</th>
      <th>company</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>reservation_status</th>
      <th>reservation_status_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>342</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/1/2015</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>737</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/1/2015</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>7</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/2/2015</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>13</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>304.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/2/2015</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>14</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>240.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.0</td>
      <td>0</td>
      <td>1</td>
      <td>Check-Out</td>
      <td>7/3/2015</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>




```python
data2= data[data['is_canceled']==0]
```


```python
plt.figure(figsize=(12,8))
plt.title('Price of Room per night & per person', fontsize=16)
sns.boxplot(x='reserved_room_type', y='adr', data=data2, hue='hotel')
plt.xlabel('Type of Room', fontsize=12)
plt.ylabel('Price in [EUR]', fontsize=12)

```




    Text(0, 0.5, 'Price in [EUR]')




    
![png](output_30_1.png)
    



```python
resort.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>arrival_date_week_number</th>
      <th>arrival_date_day_of_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>...</th>
      <th>deposit_type</th>
      <th>agent</th>
      <th>company</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>reservation_status</th>
      <th>reservation_status_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>342</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/1/2015</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>737</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/1/2015</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>7</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/2/2015</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>13</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>304.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/2/2015</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>14</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>240.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.0</td>
      <td>0</td>
      <td>1</td>
      <td>Check-Out</td>
      <td>7/3/2015</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>




```python
resort_price_per_month=resort.groupby('arrival_date_month')['adr'].mean().reset_index().sort_values(by='adr')

```


```python
city_price_per_month =city.groupby('arrival_date_month')['adr'].mean().reset_index().sort_values(by='adr')
```

**Merge resort and city hotel data**


```python
import sort_dataframeby_monthorweek as sd
hotels_prices_per_month = resort_price_per_month.merge(city_price_per_month, on='arrival_date_month')
hotels_prices_per_month.columns=['months', 'price for resort', 'price for city']
hotels_prices_per_month
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>months</th>
      <th>price for resort</th>
      <th>price for city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>November</td>
      <td>48.706289</td>
      <td>86.946592</td>
    </tr>
    <tr>
      <th>1</th>
      <td>January</td>
      <td>48.761125</td>
      <td>82.330983</td>
    </tr>
    <tr>
      <th>2</th>
      <td>February</td>
      <td>54.147478</td>
      <td>86.520062</td>
    </tr>
    <tr>
      <th>3</th>
      <td>March</td>
      <td>57.056838</td>
      <td>90.658533</td>
    </tr>
    <tr>
      <th>4</th>
      <td>October</td>
      <td>61.775449</td>
      <td>102.004672</td>
    </tr>
    <tr>
      <th>5</th>
      <td>December</td>
      <td>68.410104</td>
      <td>88.401855</td>
    </tr>
    <tr>
      <th>6</th>
      <td>April</td>
      <td>75.867816</td>
      <td>111.962267</td>
    </tr>
    <tr>
      <th>7</th>
      <td>May</td>
      <td>76.657558</td>
      <td>120.669827</td>
    </tr>
    <tr>
      <th>8</th>
      <td>September</td>
      <td>96.416860</td>
      <td>112.776582</td>
    </tr>
    <tr>
      <th>9</th>
      <td>June</td>
      <td>107.974850</td>
      <td>117.874360</td>
    </tr>
    <tr>
      <th>10</th>
      <td>July</td>
      <td>150.122528</td>
      <td>115.818019</td>
    </tr>
    <tr>
      <th>11</th>
      <td>August</td>
      <td>181.205892</td>
      <td>118.674598</td>
    </tr>
  </tbody>
</table>
</div>




```python
hotels_prices_per_month=sd.Sort_Dataframeby_Month(hotels_prices_per_month,'months')
hotels_prices_per_month
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>months</th>
      <th>price for resort</th>
      <th>price for city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>January</td>
      <td>48.761125</td>
      <td>82.330983</td>
    </tr>
    <tr>
      <th>1</th>
      <td>February</td>
      <td>54.147478</td>
      <td>86.520062</td>
    </tr>
    <tr>
      <th>2</th>
      <td>March</td>
      <td>57.056838</td>
      <td>90.658533</td>
    </tr>
    <tr>
      <th>3</th>
      <td>April</td>
      <td>75.867816</td>
      <td>111.962267</td>
    </tr>
    <tr>
      <th>4</th>
      <td>May</td>
      <td>76.657558</td>
      <td>120.669827</td>
    </tr>
    <tr>
      <th>5</th>
      <td>June</td>
      <td>107.974850</td>
      <td>117.874360</td>
    </tr>
    <tr>
      <th>6</th>
      <td>July</td>
      <td>150.122528</td>
      <td>115.818019</td>
    </tr>
    <tr>
      <th>7</th>
      <td>August</td>
      <td>181.205892</td>
      <td>118.674598</td>
    </tr>
    <tr>
      <th>8</th>
      <td>September</td>
      <td>96.416860</td>
      <td>112.776582</td>
    </tr>
    <tr>
      <th>9</th>
      <td>October</td>
      <td>61.775449</td>
      <td>102.004672</td>
    </tr>
    <tr>
      <th>10</th>
      <td>November</td>
      <td>48.706289</td>
      <td>86.946592</td>
    </tr>
    <tr>
      <th>11</th>
      <td>December</td>
      <td>68.410104</td>
      <td>88.401855</td>
    </tr>
  </tbody>
</table>
</div>




```python
px.line(hotels_prices_per_month, x='months',y=['price for resort','price for city'],
        title='Room price per night over the year')
```


<div>                            <div id="aaaf9624-de50-4084-80ce-5b3a37db4859" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("aaaf9624-de50-4084-80ce-5b3a37db4859")) {                    Plotly.newPlot(                        "aaaf9624-de50-4084-80ce-5b3a37db4859",                        [{"hovertemplate":"variable=price for resort<br>months=%{x}<br>value=%{y}<extra></extra>","legendgroup":"price for resort","line":{"color":"#636efa","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines","name":"price for resort","orientation":"v","showlegend":true,"x":["January","February","March","April","May","June","July","August","September","October","November","December"],"xaxis":"x","y":[48.761125401929256,54.14747833622184,57.056837806301054,75.86781568627451,76.65755818540434,107.97485027000491,150.1225278928913,181.20589192508442,96.41686013320647,61.77544854368932,48.706288607594935,68.41010427010924],"yaxis":"y","type":"scatter"},{"hovertemplate":"variable=price for city<br>months=%{x}<br>value=%{y}<extra></extra>","legendgroup":"price for city","line":{"color":"#EF553B","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines","name":"price for city","orientation":"v","showlegend":true,"x":["January","February","March","April","May","June","July","August","September","October","November","December"],"xaxis":"x","y":[82.33098265895954,86.52006227466406,90.65853297110398,111.9622668329177,120.66982705779336,117.87435979807252,115.81801886792452,118.67459847214458,112.77658183516226,102.00467175219603,86.94659192825111,88.4018552797644],"yaxis":"y","type":"scatter"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"months"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"value"}},"legend":{"title":{"text":"variable"},"tracegroupgap":0},"title":{"text":"Room price per night over the year"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('aaaf9624-de50-4084-80ce-5b3a37db4859');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


This clearly shows that the prices in the Resort hotel are much higher during the summer (no surprise here).
The price of the city hotel varies less and is most expensive during spring and autumn


```python
data['market_segment'].unique()
```




    array(['Direct', 'Corporate', 'Online TA', 'Offline TA/TO',
           'Complementary', 'Groups', 'Undefined', 'Aviation'], dtype=object)




```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>arrival_date_week_number</th>
      <th>arrival_date_day_of_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>...</th>
      <th>deposit_type</th>
      <th>agent</th>
      <th>company</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>reservation_status</th>
      <th>reservation_status_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>342</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/1/2015</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>737</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/1/2015</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>7</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/2/2015</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>13</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>304.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/2/2015</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>14</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>240.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.0</td>
      <td>0</td>
      <td>1</td>
      <td>Check-Out</td>
      <td>7/3/2015</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>




```python
order = data2.groupby('market_segment')['stays_in_weekend_nights'].mean().sort_values(ascending=False).index
```


```python
plt.figure(figsize=(12,8))
sns.barplot(x='market_segment',y='stays_in_weekend_nights', data = data2, hue='hotel', order=order)
```




    <AxesSubplot:xlabel='market_segment', ylabel='stays_in_weekend_nights'>




    
![png](output_42_1.png)
    


### Bookings by market segmant


```python
data.groupby('market_segment').count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>arrival_date_week_number</th>
      <th>arrival_date_day_of_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>...</th>
      <th>deposit_type</th>
      <th>agent</th>
      <th>company</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>reservation_status</th>
      <th>reservation_status_date</th>
    </tr>
    <tr>
      <th>market_segment</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Aviation</th>
      <td>235</td>
      <td>235</td>
      <td>235</td>
      <td>235</td>
      <td>235</td>
      <td>235</td>
      <td>235</td>
      <td>235</td>
      <td>235</td>
      <td>235</td>
      <td>...</td>
      <td>235</td>
      <td>235</td>
      <td>235</td>
      <td>235</td>
      <td>235</td>
      <td>235</td>
      <td>235</td>
      <td>235</td>
      <td>235</td>
      <td>235</td>
    </tr>
    <tr>
      <th>Complementary</th>
      <td>728</td>
      <td>728</td>
      <td>728</td>
      <td>728</td>
      <td>728</td>
      <td>728</td>
      <td>728</td>
      <td>728</td>
      <td>728</td>
      <td>728</td>
      <td>...</td>
      <td>728</td>
      <td>728</td>
      <td>728</td>
      <td>728</td>
      <td>728</td>
      <td>728</td>
      <td>728</td>
      <td>728</td>
      <td>728</td>
      <td>728</td>
    </tr>
    <tr>
      <th>Corporate</th>
      <td>5282</td>
      <td>5282</td>
      <td>5282</td>
      <td>5282</td>
      <td>5282</td>
      <td>5282</td>
      <td>5282</td>
      <td>5282</td>
      <td>5282</td>
      <td>5282</td>
      <td>...</td>
      <td>5282</td>
      <td>5282</td>
      <td>5282</td>
      <td>5282</td>
      <td>5282</td>
      <td>5282</td>
      <td>5282</td>
      <td>5282</td>
      <td>5282</td>
      <td>5282</td>
    </tr>
    <tr>
      <th>Direct</th>
      <td>12582</td>
      <td>12582</td>
      <td>12582</td>
      <td>12582</td>
      <td>12582</td>
      <td>12582</td>
      <td>12582</td>
      <td>12582</td>
      <td>12582</td>
      <td>12582</td>
      <td>...</td>
      <td>12582</td>
      <td>12582</td>
      <td>12582</td>
      <td>12582</td>
      <td>12582</td>
      <td>12582</td>
      <td>12582</td>
      <td>12582</td>
      <td>12582</td>
      <td>12582</td>
    </tr>
    <tr>
      <th>Groups</th>
      <td>19791</td>
      <td>19791</td>
      <td>19791</td>
      <td>19791</td>
      <td>19791</td>
      <td>19791</td>
      <td>19791</td>
      <td>19791</td>
      <td>19791</td>
      <td>19791</td>
      <td>...</td>
      <td>19791</td>
      <td>19791</td>
      <td>19791</td>
      <td>19791</td>
      <td>19791</td>
      <td>19791</td>
      <td>19791</td>
      <td>19791</td>
      <td>19791</td>
      <td>19791</td>
    </tr>
    <tr>
      <th>Offline TA/TO</th>
      <td>24182</td>
      <td>24182</td>
      <td>24182</td>
      <td>24182</td>
      <td>24182</td>
      <td>24182</td>
      <td>24182</td>
      <td>24182</td>
      <td>24182</td>
      <td>24182</td>
      <td>...</td>
      <td>24182</td>
      <td>24182</td>
      <td>24182</td>
      <td>24182</td>
      <td>24182</td>
      <td>24182</td>
      <td>24182</td>
      <td>24182</td>
      <td>24182</td>
      <td>24182</td>
    </tr>
    <tr>
      <th>Online TA</th>
      <td>56408</td>
      <td>56408</td>
      <td>56408</td>
      <td>56408</td>
      <td>56408</td>
      <td>56408</td>
      <td>56408</td>
      <td>56408</td>
      <td>56408</td>
      <td>56408</td>
      <td>...</td>
      <td>56408</td>
      <td>56408</td>
      <td>56408</td>
      <td>56408</td>
      <td>56408</td>
      <td>56408</td>
      <td>56408</td>
      <td>56408</td>
      <td>56408</td>
      <td>56408</td>
    </tr>
    <tr>
      <th>Undefined</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 31 columns</p>
</div>




```python
segments =df['market_segment'].value_counts()
```


```python
fig = px.pie(segments,values=segments.values,names=segments.index,title='Bookings by market segments' ,template='seaborn')
fig.update_traces(rotation=-90, textinfo="percent+label")

```


<div>                            <div id="a2cfb4b0-c442-4fe2-8d46-92cdbce44dcd" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("a2cfb4b0-c442-4fe2-8d46-92cdbce44dcd")) {                    Plotly.newPlot(                        "a2cfb4b0-c442-4fe2-8d46-92cdbce44dcd",                        [{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"hovertemplate":"index=%{label}<br>value=%{value}<extra></extra>","labels":["Online TA","Offline TA/TO","Groups","Direct","Corporate","Complementary","Aviation","Undefined"],"legendgroup":"","name":"","showlegend":true,"values":[56477,24219,19811,12606,5295,743,237,2],"type":"pie","rotation":-90,"textinfo":"percent+label"}],                        {"template":{"data":{"barpolar":[{"marker":{"line":{"color":"rgb(234,234,242)","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"rgb(36,36,36)"},"error_y":{"color":"rgb(36,36,36)"},"marker":{"line":{"color":"rgb(234,234,242)","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"rgb(36,36,36)","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"rgb(36,36,36)"},"baxis":{"endlinecolor":"rgb(36,36,36)","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"rgb(36,36,36)"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"tickcolor":"rgb(36,36,36)","ticklen":8,"ticks":"outside","tickwidth":2},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"tickcolor":"rgb(36,36,36)","ticklen":8,"ticks":"outside","tickwidth":2},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"tickcolor":"rgb(36,36,36)","ticklen":8,"ticks":"outside","tickwidth":2},"colorscale":[[0.0,"rgb(2,4,25)"],[0.06274509803921569,"rgb(24,15,41)"],[0.12549019607843137,"rgb(47,23,57)"],[0.18823529411764706,"rgb(71,28,72)"],[0.25098039215686274,"rgb(97,30,82)"],[0.3137254901960784,"rgb(123,30,89)"],[0.3764705882352941,"rgb(150,27,91)"],[0.4392156862745098,"rgb(177,22,88)"],[0.5019607843137255,"rgb(203,26,79)"],[0.5647058823529412,"rgb(223,47,67)"],[0.6274509803921569,"rgb(236,76,61)"],[0.6901960784313725,"rgb(242,107,73)"],[0.7529411764705882,"rgb(244,135,95)"],[0.8156862745098039,"rgb(245,162,122)"],[0.8784313725490196,"rgb(246,188,153)"],[0.9411764705882353,"rgb(247,212,187)"],[1.0,"rgb(250,234,220)"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"tickcolor":"rgb(36,36,36)","ticklen":8,"ticks":"outside","tickwidth":2},"colorscale":[[0.0,"rgb(2,4,25)"],[0.06274509803921569,"rgb(24,15,41)"],[0.12549019607843137,"rgb(47,23,57)"],[0.18823529411764706,"rgb(71,28,72)"],[0.25098039215686274,"rgb(97,30,82)"],[0.3137254901960784,"rgb(123,30,89)"],[0.3764705882352941,"rgb(150,27,91)"],[0.4392156862745098,"rgb(177,22,88)"],[0.5019607843137255,"rgb(203,26,79)"],[0.5647058823529412,"rgb(223,47,67)"],[0.6274509803921569,"rgb(236,76,61)"],[0.6901960784313725,"rgb(242,107,73)"],[0.7529411764705882,"rgb(244,135,95)"],[0.8156862745098039,"rgb(245,162,122)"],[0.8784313725490196,"rgb(246,188,153)"],[0.9411764705882353,"rgb(247,212,187)"],[1.0,"rgb(250,234,220)"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":0,"tickcolor":"rgb(36,36,36)","ticklen":8,"ticks":"outside","tickwidth":2},"colorscale":[[0.0,"rgb(2,4,25)"],[0.06274509803921569,"rgb(24,15,41)"],[0.12549019607843137,"rgb(47,23,57)"],[0.18823529411764706,"rgb(71,28,72)"],[0.25098039215686274,"rgb(97,30,82)"],[0.3137254901960784,"rgb(123,30,89)"],[0.3764705882352941,"rgb(150,27,91)"],[0.4392156862745098,"rgb(177,22,88)"],[0.5019607843137255,"rgb(203,26,79)"],[0.5647058823529412,"rgb(223,47,67)"],[0.6274509803921569,"rgb(236,76,61)"],[0.6901960784313725,"rgb(242,107,73)"],[0.7529411764705882,"rgb(244,135,95)"],[0.8156862745098039,"rgb(245,162,122)"],[0.8784313725490196,"rgb(246,188,153)"],[0.9411764705882353,"rgb(247,212,187)"],[1.0,"rgb(250,234,220)"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"tickcolor":"rgb(36,36,36)","ticklen":8,"ticks":"outside","tickwidth":2},"colorscale":[[0.0,"rgb(2,4,25)"],[0.06274509803921569,"rgb(24,15,41)"],[0.12549019607843137,"rgb(47,23,57)"],[0.18823529411764706,"rgb(71,28,72)"],[0.25098039215686274,"rgb(97,30,82)"],[0.3137254901960784,"rgb(123,30,89)"],[0.3764705882352941,"rgb(150,27,91)"],[0.4392156862745098,"rgb(177,22,88)"],[0.5019607843137255,"rgb(203,26,79)"],[0.5647058823529412,"rgb(223,47,67)"],[0.6274509803921569,"rgb(236,76,61)"],[0.6901960784313725,"rgb(242,107,73)"],[0.7529411764705882,"rgb(244,135,95)"],[0.8156862745098039,"rgb(245,162,122)"],[0.8784313725490196,"rgb(246,188,153)"],[0.9411764705882353,"rgb(247,212,187)"],[1.0,"rgb(250,234,220)"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"tickcolor":"rgb(36,36,36)","ticklen":8,"ticks":"outside","tickwidth":2},"colorscale":[[0.0,"rgb(2,4,25)"],[0.06274509803921569,"rgb(24,15,41)"],[0.12549019607843137,"rgb(47,23,57)"],[0.18823529411764706,"rgb(71,28,72)"],[0.25098039215686274,"rgb(97,30,82)"],[0.3137254901960784,"rgb(123,30,89)"],[0.3764705882352941,"rgb(150,27,91)"],[0.4392156862745098,"rgb(177,22,88)"],[0.5019607843137255,"rgb(203,26,79)"],[0.5647058823529412,"rgb(223,47,67)"],[0.6274509803921569,"rgb(236,76,61)"],[0.6901960784313725,"rgb(242,107,73)"],[0.7529411764705882,"rgb(244,135,95)"],[0.8156862745098039,"rgb(245,162,122)"],[0.8784313725490196,"rgb(246,188,153)"],[0.9411764705882353,"rgb(247,212,187)"],[1.0,"rgb(250,234,220)"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"tickcolor":"rgb(36,36,36)","ticklen":8,"ticks":"outside","tickwidth":2},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"tickcolor":"rgb(36,36,36)","ticklen":8,"ticks":"outside","tickwidth":2}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"tickcolor":"rgb(36,36,36)","ticklen":8,"ticks":"outside","tickwidth":2}},"marker":{"colorbar":{"outlinewidth":0,"tickcolor":"rgb(36,36,36)","ticklen":8,"ticks":"outside","tickwidth":2}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"tickcolor":"rgb(36,36,36)","ticklen":8,"ticks":"outside","tickwidth":2}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"tickcolor":"rgb(36,36,36)","ticklen":8,"ticks":"outside","tickwidth":2}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"tickcolor":"rgb(36,36,36)","ticklen":8,"ticks":"outside","tickwidth":2}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"tickcolor":"rgb(36,36,36)","ticklen":8,"ticks":"outside","tickwidth":2}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"tickcolor":"rgb(36,36,36)","ticklen":8,"ticks":"outside","tickwidth":2}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"tickcolor":"rgb(36,36,36)","ticklen":8,"ticks":"outside","tickwidth":2}},"type":"scatterpolar"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"tickcolor":"rgb(36,36,36)","ticklen":8,"ticks":"outside","tickwidth":2}},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"tickcolor":"rgb(36,36,36)","ticklen":8,"ticks":"outside","tickwidth":2}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"tickcolor":"rgb(36,36,36)","ticklen":8,"ticks":"outside","tickwidth":2},"colorscale":[[0.0,"rgb(2,4,25)"],[0.06274509803921569,"rgb(24,15,41)"],[0.12549019607843137,"rgb(47,23,57)"],[0.18823529411764706,"rgb(71,28,72)"],[0.25098039215686274,"rgb(97,30,82)"],[0.3137254901960784,"rgb(123,30,89)"],[0.3764705882352941,"rgb(150,27,91)"],[0.4392156862745098,"rgb(177,22,88)"],[0.5019607843137255,"rgb(203,26,79)"],[0.5647058823529412,"rgb(223,47,67)"],[0.6274509803921569,"rgb(236,76,61)"],[0.6901960784313725,"rgb(242,107,73)"],[0.7529411764705882,"rgb(244,135,95)"],[0.8156862745098039,"rgb(245,162,122)"],[0.8784313725490196,"rgb(246,188,153)"],[0.9411764705882353,"rgb(247,212,187)"],[1.0,"rgb(250,234,220)"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"rgb(231,231,240)"},"line":{"color":"white"}},"header":{"fill":{"color":"rgb(183,183,191)"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"rgb(67,103,167)"},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"tickcolor":"rgb(36,36,36)","ticklen":8,"ticks":"outside","tickwidth":2}},"colorscale":{"sequential":[[0.0,"rgb(2,4,25)"],[0.06274509803921569,"rgb(24,15,41)"],[0.12549019607843137,"rgb(47,23,57)"],[0.18823529411764706,"rgb(71,28,72)"],[0.25098039215686274,"rgb(97,30,82)"],[0.3137254901960784,"rgb(123,30,89)"],[0.3764705882352941,"rgb(150,27,91)"],[0.4392156862745098,"rgb(177,22,88)"],[0.5019607843137255,"rgb(203,26,79)"],[0.5647058823529412,"rgb(223,47,67)"],[0.6274509803921569,"rgb(236,76,61)"],[0.6901960784313725,"rgb(242,107,73)"],[0.7529411764705882,"rgb(244,135,95)"],[0.8156862745098039,"rgb(245,162,122)"],[0.8784313725490196,"rgb(246,188,153)"],[0.9411764705882353,"rgb(247,212,187)"],[1.0,"rgb(250,234,220)"]],"sequentialminus":[[0.0,"rgb(2,4,25)"],[0.06274509803921569,"rgb(24,15,41)"],[0.12549019607843137,"rgb(47,23,57)"],[0.18823529411764706,"rgb(71,28,72)"],[0.25098039215686274,"rgb(97,30,82)"],[0.3137254901960784,"rgb(123,30,89)"],[0.3764705882352941,"rgb(150,27,91)"],[0.4392156862745098,"rgb(177,22,88)"],[0.5019607843137255,"rgb(203,26,79)"],[0.5647058823529412,"rgb(223,47,67)"],[0.6274509803921569,"rgb(236,76,61)"],[0.6901960784313725,"rgb(242,107,73)"],[0.7529411764705882,"rgb(244,135,95)"],[0.8156862745098039,"rgb(245,162,122)"],[0.8784313725490196,"rgb(246,188,153)"],[0.9411764705882353,"rgb(247,212,187)"],[1.0,"rgb(250,234,220)"]]},"colorway":["rgb(76,114,176)","rgb(221,132,82)","rgb(85,168,104)","rgb(196,78,82)","rgb(129,114,179)","rgb(147,120,96)","rgb(218,139,195)","rgb(140,140,140)","rgb(204,185,116)","rgb(100,181,205)"],"font":{"color":"rgb(36,36,36)"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"rgb(234,234,242)","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","paper_bgcolor":"white","plot_bgcolor":"rgb(234,234,242)","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","showgrid":true,"ticks":""},"bgcolor":"rgb(234,234,242)","radialaxis":{"gridcolor":"white","linecolor":"white","showgrid":true,"ticks":""}},"scene":{"xaxis":{"backgroundcolor":"rgb(234,234,242)","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"showgrid":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"rgb(234,234,242)","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"showgrid":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"rgb(234,234,242)","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"showgrid":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"fillcolor":"rgb(67,103,167)","line":{"width":0},"opacity":0.5},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","showgrid":true,"ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","showgrid":true,"ticks":""},"bgcolor":"rgb(234,234,242)","caxis":{"gridcolor":"white","linecolor":"white","showgrid":true,"ticks":""}},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","showgrid":true,"ticks":"","title":{"standoff":15},"zerolinecolor":"white"},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","showgrid":true,"ticks":"","title":{"standoff":15},"zerolinecolor":"white"}}},"legend":{"tracegroupgap":0},"title":{"text":"Bookings by market segments"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('a2cfb4b0-c442-4fe2-8d46-92cdbce44dcd');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
data['meal'].value_counts()
```




    BB           92236
    HB           14458
    SC           10549
    Undefined     1169
    FB             798
    Name: meal, dtype: int64




```python
px.pie(values=data['meal'].value_counts(), names=data['meal'].value_counts().index, hole=.5,
       title='Meals Chosen with reservations').update_traces( textinfo="percent+label" )
```


<div>                            <div id="805d0d88-e582-4835-af91-f2b3218d0407" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("805d0d88-e582-4835-af91-f2b3218d0407")) {                    Plotly.newPlot(                        "805d0d88-e582-4835-af91-f2b3218d0407",                        [{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"hole":0.5,"hovertemplate":"label=%{label}<br>value=%{value}<extra></extra>","labels":["BB","HB","SC","Undefined","FB"],"legendgroup":"","name":"","showlegend":true,"values":[92236,14458,10549,1169,798],"type":"pie","textinfo":"percent+label"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"legend":{"tracegroupgap":0},"title":{"text":"Meals Chosen with reservations"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('805d0d88-e582-4835-af91-f2b3218d0407');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


It is clear that guests prefer to have only the breakfast meal in the hotel

### Analysis of special Requests 


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>arrival_date_week_number</th>
      <th>arrival_date_day_of_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>...</th>
      <th>deposit_type</th>
      <th>agent</th>
      <th>company</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>reservation_status</th>
      <th>reservation_status_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>342</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/1/2015</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>737</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/1/2015</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>7</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/2/2015</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>13</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>304.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/2/2015</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>14</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>240.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.0</td>
      <td>0</td>
      <td>1</td>
      <td>Check-Out</td>
      <td>7/3/2015</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>




```python
#numbers of special requests
sns.countplot(data['total_of_special_requests'])
```

    C:\Users\RO2YA\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning:
    
    Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
    
    




    <AxesSubplot:xlabel='total_of_special_requests', ylabel='count'>




    
![png](output_52_2.png)
    



```python
# to get the percent of special requests 
data[data['total_of_special_requests']>=1]['total_of_special_requests'].count()/len(data) *100
```




    41.11148393591142



Based on the previous, 41.11 % of guests have spacial requests


```python
pivot_spe_req = data.groupby(['total_of_special_requests', 'is_canceled']).agg({'total_of_special_requests':'count'}).rename(columns={'total_of_special_requests':'count'}).unstack()
pivot_spe_req
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">count</th>
    </tr>
    <tr>
      <th>is_canceled</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>total_of_special_requests</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>36667</td>
      <td>33534</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25867</td>
      <td>7316</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10086</td>
      <td>2866</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2049</td>
      <td>445</td>
    </tr>
    <tr>
      <th>4</th>
      <td>304</td>
      <td>36</td>
    </tr>
    <tr>
      <th>5</th>
      <td>38</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
pivot_spe_req.plot(kind='bar' )
```




    <AxesSubplot:xlabel='total_of_special_requests'>




    
![png](output_56_1.png)
    


Almost half of bookings with zero special requests have been canceled

### What are the most busy months? 

**First we get the busy month of both hotels compined and then seperated**


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>arrival_date_week_number</th>
      <th>arrival_date_day_of_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>...</th>
      <th>deposit_type</th>
      <th>agent</th>
      <th>company</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>reservation_status</th>
      <th>reservation_status_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>342</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/1/2015</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>737</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/1/2015</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>7</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/2/2015</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>13</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>304.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/2/2015</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>14</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>240.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.0</td>
      <td>0</td>
      <td>1</td>
      <td>Check-Out</td>
      <td>7/3/2015</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>




```python
num_guests_per_month = data2.groupby('arrival_date_month').count()['arrival_date_day_of_month'].reset_index().rename(columns={'arrival_date_day_of_month':'num of guests', 'arrival_date_month':'month'})
```


```python
num_guests_per_month = sd.Sort_Dataframeby_Month(num_guests_per_month,'month')
```


```python
num_guests_per_month
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>month</th>
      <th>num of guests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>January</td>
      <td>4115</td>
    </tr>
    <tr>
      <th>1</th>
      <td>February</td>
      <td>5359</td>
    </tr>
    <tr>
      <th>2</th>
      <td>March</td>
      <td>6620</td>
    </tr>
    <tr>
      <th>3</th>
      <td>April</td>
      <td>6560</td>
    </tr>
    <tr>
      <th>4</th>
      <td>May</td>
      <td>7103</td>
    </tr>
    <tr>
      <th>5</th>
      <td>June</td>
      <td>6395</td>
    </tr>
    <tr>
      <th>6</th>
      <td>July</td>
      <td>7907</td>
    </tr>
    <tr>
      <th>7</th>
      <td>August</td>
      <td>8624</td>
    </tr>
    <tr>
      <th>8</th>
      <td>September</td>
      <td>6385</td>
    </tr>
    <tr>
      <th>9</th>
      <td>October</td>
      <td>6901</td>
    </tr>
    <tr>
      <th>10</th>
      <td>November</td>
      <td>4651</td>
    </tr>
    <tr>
      <th>11</th>
      <td>December</td>
      <td>4391</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(12,8))
sns.lineplot(x='month', y='num of guests', data= num_guests_per_month)

```




    <AxesSubplot:xlabel='month', ylabel='num of guests'>




    
![png](output_64_1.png)
    


The most busy months for both hotels are July and August


```python
guests_per_month_2hotels =data2.groupby(['hotel', 'arrival_date_month']).agg({'arrival_date_month':'count'}).rename(columns={'arrival_date_month':'num of guests'}).unstack()
```


```python
guests_per_month_2hotels
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="12" halign="left">num of guests</th>
    </tr>
    <tr>
      <th>arrival_date_month</th>
      <th>April</th>
      <th>August</th>
      <th>December</th>
      <th>February</th>
      <th>January</th>
      <th>July</th>
      <th>June</th>
      <th>March</th>
      <th>May</th>
      <th>November</th>
      <th>October</th>
      <th>September</th>
    </tr>
    <tr>
      <th>hotel</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>City Hotel</th>
      <td>4010</td>
      <td>5367</td>
      <td>2377</td>
      <td>3051</td>
      <td>2249</td>
      <td>4770</td>
      <td>4358</td>
      <td>4049</td>
      <td>4568</td>
      <td>2676</td>
      <td>4326</td>
      <td>4283</td>
    </tr>
    <tr>
      <th>Resort Hotel</th>
      <td>2550</td>
      <td>3257</td>
      <td>2014</td>
      <td>2308</td>
      <td>1866</td>
      <td>3137</td>
      <td>2037</td>
      <td>2571</td>
      <td>2535</td>
      <td>1975</td>
      <td>2575</td>
      <td>2102</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(guests_per_month_2hotels.shape)
```

    (2, 12)
    


```python
guests_resort = resort['arrival_date_month'].value_counts().reset_index()
guests_resort.columns=['months','no_of_guests']
guests_resort=sd.Sort_Dataframeby_Month(guests_resort,'months')
```


```python
guests_resort
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>months</th>
      <th>no_of_guests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>January</td>
      <td>1866</td>
    </tr>
    <tr>
      <th>1</th>
      <td>February</td>
      <td>2308</td>
    </tr>
    <tr>
      <th>2</th>
      <td>March</td>
      <td>2571</td>
    </tr>
    <tr>
      <th>3</th>
      <td>April</td>
      <td>2550</td>
    </tr>
    <tr>
      <th>4</th>
      <td>May</td>
      <td>2535</td>
    </tr>
    <tr>
      <th>5</th>
      <td>June</td>
      <td>2037</td>
    </tr>
    <tr>
      <th>6</th>
      <td>July</td>
      <td>3137</td>
    </tr>
    <tr>
      <th>7</th>
      <td>August</td>
      <td>3257</td>
    </tr>
    <tr>
      <th>8</th>
      <td>September</td>
      <td>2102</td>
    </tr>
    <tr>
      <th>9</th>
      <td>October</td>
      <td>2575</td>
    </tr>
    <tr>
      <th>10</th>
      <td>November</td>
      <td>1975</td>
    </tr>
    <tr>
      <th>11</th>
      <td>December</td>
      <td>2014</td>
    </tr>
  </tbody>
</table>
</div>




```python
guests_city = city['arrival_date_month'].value_counts().reset_index()
guests_city.columns=['months','no_of_guests']
guests_city=sd.Sort_Dataframeby_Month(guests_city,'months')
guests_city
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>months</th>
      <th>no_of_guests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>January</td>
      <td>2249</td>
    </tr>
    <tr>
      <th>1</th>
      <td>February</td>
      <td>3051</td>
    </tr>
    <tr>
      <th>2</th>
      <td>March</td>
      <td>4049</td>
    </tr>
    <tr>
      <th>3</th>
      <td>April</td>
      <td>4010</td>
    </tr>
    <tr>
      <th>4</th>
      <td>May</td>
      <td>4568</td>
    </tr>
    <tr>
      <th>5</th>
      <td>June</td>
      <td>4358</td>
    </tr>
    <tr>
      <th>6</th>
      <td>July</td>
      <td>4770</td>
    </tr>
    <tr>
      <th>7</th>
      <td>August</td>
      <td>5367</td>
    </tr>
    <tr>
      <th>8</th>
      <td>September</td>
      <td>4283</td>
    </tr>
    <tr>
      <th>9</th>
      <td>October</td>
      <td>4326</td>
    </tr>
    <tr>
      <th>10</th>
      <td>November</td>
      <td>2676</td>
    </tr>
    <tr>
      <th>11</th>
      <td>December</td>
      <td>2377</td>
    </tr>
  </tbody>
</table>
</div>




```python
guests = guests_resort.merge(guests_city,on='months')
guests.columns=['months','no_guests_resort','no_guests_city']
guests
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>months</th>
      <th>no_guests_resort</th>
      <th>no_guests_city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>January</td>
      <td>1866</td>
      <td>2249</td>
    </tr>
    <tr>
      <th>1</th>
      <td>February</td>
      <td>2308</td>
      <td>3051</td>
    </tr>
    <tr>
      <th>2</th>
      <td>March</td>
      <td>2571</td>
      <td>4049</td>
    </tr>
    <tr>
      <th>3</th>
      <td>April</td>
      <td>2550</td>
      <td>4010</td>
    </tr>
    <tr>
      <th>4</th>
      <td>May</td>
      <td>2535</td>
      <td>4568</td>
    </tr>
    <tr>
      <th>5</th>
      <td>June</td>
      <td>2037</td>
      <td>4358</td>
    </tr>
    <tr>
      <th>6</th>
      <td>July</td>
      <td>3137</td>
      <td>4770</td>
    </tr>
    <tr>
      <th>7</th>
      <td>August</td>
      <td>3257</td>
      <td>5367</td>
    </tr>
    <tr>
      <th>8</th>
      <td>September</td>
      <td>2102</td>
      <td>4283</td>
    </tr>
    <tr>
      <th>9</th>
      <td>October</td>
      <td>2575</td>
      <td>4326</td>
    </tr>
    <tr>
      <th>10</th>
      <td>November</td>
      <td>1975</td>
      <td>2676</td>
    </tr>
    <tr>
      <th>11</th>
      <td>December</td>
      <td>2014</td>
      <td>2377</td>
    </tr>
  </tbody>
</table>
</div>




```python
px.line(hotels_prices_per_month, x='months',y=['price for resort','price for city'],
        title='Room price per night over the year')
px.line(guests, x='months',y=['no_guests_resort','no_guests_city'], title='Total number of guests per month during the year')
```


<div>                            <div id="644e8dfc-3ac6-4085-82ab-4e52a0a82b4c" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("644e8dfc-3ac6-4085-82ab-4e52a0a82b4c")) {                    Plotly.newPlot(                        "644e8dfc-3ac6-4085-82ab-4e52a0a82b4c",                        [{"hovertemplate":"variable=no_guests_resort<br>months=%{x}<br>value=%{y}<extra></extra>","legendgroup":"no_guests_resort","line":{"color":"#636efa","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines","name":"no_guests_resort","orientation":"v","showlegend":true,"x":["January","February","March","April","May","June","July","August","September","October","November","December"],"xaxis":"x","y":[1866,2308,2571,2550,2535,2037,3137,3257,2102,2575,1975,2014],"yaxis":"y","type":"scatter"},{"hovertemplate":"variable=no_guests_city<br>months=%{x}<br>value=%{y}<extra></extra>","legendgroup":"no_guests_city","line":{"color":"#EF553B","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines","name":"no_guests_city","orientation":"v","showlegend":true,"x":["January","February","March","April","May","June","July","August","September","October","November","December"],"xaxis":"x","y":[2249,3051,4049,4010,4568,4358,4770,5367,4283,4326,2676,2377],"yaxis":"y","type":"scatter"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"months"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"value"}},"legend":{"title":{"text":"variable"},"tracegroupgap":0},"title":{"text":"Total number of guests per month during the year"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('644e8dfc-3ac6-4085-82ab-4e52a0a82b4c');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


City club hotel gets more guests than resort hotel and the traffic of both hotels during the year, both gets attracts morer guests in summer months specially in August

### Comparing the traffic of guests and the prices 


```python
from plotly.subplots import make_subplots
fig = make_subplots(rows=1, cols=2)

fig.add_trace(go.Scatter(x=hotels_prices_per_month['months'], y=hotels_prices_per_month['price for resort'], name='Price for resort'), row=1, col=1)
fig.add_trace(go.Scatter(x=hotels_prices_per_month['months'], y=hotels_prices_per_month['price for city'], name='Price for city'), row=1, col=1)
fig.update_xaxes(title_text='Months', row=1, col=1)
fig.update_yaxes(title_text='Price', row=1, col=1)

fig.add_trace(go.Scatter(x=guests['months'], y=guests['no_guests_resort'], name='Number of guests in resort'), row=1, col=2)
fig.add_trace(go.Scatter(x=guests['months'], y=guests['no_guests_city'], name='Number of guests in city'), row=1, col=2)
fig.update_xaxes(title_text='Months', row=1, col=2)
fig.update_yaxes(title_text='Number of guests', row=1, col=2)
```


<div>                            <div id="2d0ab60c-b7d4-49fe-af68-ccc71890bf5b" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("2d0ab60c-b7d4-49fe-af68-ccc71890bf5b")) {                    Plotly.newPlot(                        "2d0ab60c-b7d4-49fe-af68-ccc71890bf5b",                        [{"name":"Price for resort","x":["January","February","March","April","May","June","July","August","September","October","November","December"],"y":[48.761125401929256,54.14747833622184,57.056837806301054,75.86781568627451,76.65755818540434,107.97485027000491,150.1225278928913,181.20589192508442,96.41686013320647,61.77544854368932,48.706288607594935,68.41010427010924],"type":"scatter","xaxis":"x","yaxis":"y"},{"name":"Price for city","x":["January","February","March","April","May","June","July","August","September","October","November","December"],"y":[82.33098265895954,86.52006227466406,90.65853297110398,111.9622668329177,120.66982705779336,117.87435979807252,115.81801886792452,118.67459847214458,112.77658183516226,102.00467175219603,86.94659192825111,88.4018552797644],"type":"scatter","xaxis":"x","yaxis":"y"},{"name":"Number of guests in resort","x":["January","February","March","April","May","June","July","August","September","October","November","December"],"y":[1866,2308,2571,2550,2535,2037,3137,3257,2102,2575,1975,2014],"type":"scatter","xaxis":"x2","yaxis":"y2"},{"name":"Number of guests in city","x":["January","February","March","April","May","June","July","August","September","October","November","December"],"y":[2249,3051,4049,4010,4568,4358,4770,5367,4283,4326,2676,2377],"type":"scatter","xaxis":"x2","yaxis":"y2"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"xaxis":{"anchor":"y","domain":[0.0,0.45],"title":{"text":"Months"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Price"}},"xaxis2":{"anchor":"y2","domain":[0.55,1.0],"title":{"text":"Months"}},"yaxis2":{"anchor":"x2","domain":[0.0,1.0],"title":{"text":"Number of guests"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('2d0ab60c-b7d4-49fe-af68-ccc71890bf5b');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


City club attracts more guests during automn and summer months while the prices rises because of high demand at these times 
and resort hotel attracts more customers during July and August and gets lower guests at June and September while the prices becomes hightest during July and August Because of high demand 


```python
data['is_canceled'].unique()
```




    array([0, 1], dtype=int64)




```python
valid_bookings = data[data['is_canceled']==0]
```


```python
valid_bookings['total_nights']=valid_bookings['stays_in_weekend_nights'] +valid_bookings['stays_in_week_nights']
```


```python
valid_bookings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>arrival_date_week_number</th>
      <th>arrival_date_day_of_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>...</th>
      <th>agent</th>
      <th>company</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>reservation_status</th>
      <th>reservation_status_date</th>
      <th>total_nights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>342</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/1/2015</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>737</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/1/2015</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>7</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/2/2015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>13</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>304.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/2/2015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>14</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>240.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.0</td>
      <td>0</td>
      <td>1</td>
      <td>Check-Out</td>
      <td>7/3/2015</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>




```python
stay_df = valid_bookings.groupby(['total_nights', 'hotel']).agg('count').reset_index()
stay_df =stay_df.iloc[:,0:3]
stay_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_nights</th>
      <th>hotel</th>
      <th>is_canceled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>City Hotel</td>
      <td>251</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Resort Hotel</td>
      <td>371</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>City Hotel</td>
      <td>9155</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Resort Hotel</td>
      <td>6579</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>City Hotel</td>
      <td>10983</td>
    </tr>
  </tbody>
</table>
</div>




```python
# to rename the column 'is_canceled'
stay_df=stay_df.rename(columns={'is_canceled':'number_of_stays'})
stay_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_nights</th>
      <th>hotel</th>
      <th>number_of_stays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>City Hotel</td>
      <td>251</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Resort Hotel</td>
      <td>371</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>City Hotel</td>
      <td>9155</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Resort Hotel</td>
      <td>6579</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>City Hotel</td>
      <td>10983</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(12,8))
sns.barplot(x='total_nights', y='number_of_stays', data=stay_df, hue='hotel' ,hue_order=['City Hotel','Resort Hotel'], palette='coolwarm')
plt.legend(loc='upper right', fontsize=14)

```




    <matplotlib.legend.Legend at 0x1fe5ca3c880>




    
![png](output_84_1.png)
    


For the city hotel there is a clear preference for 1-4 nights. <br>
For the resort hotel, 1-4 nights are also often booked, but 7 nights also stand out as being very popular.


```python

```


```python

```
