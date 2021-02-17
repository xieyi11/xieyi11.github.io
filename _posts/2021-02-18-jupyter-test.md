---
layout: post
title:  "decision tree / random forest by skicit-learn"
date:   2021-02-17 11:23:32
categories: ML
---


# 一、读入数据


```python
import pandas as pd
df = pd.read_csv('diamonds.csv',index_col=0)
print(f'前几行数据：\n{df.head()}')
print(f'数据大小：\n{df.shape}')
print(f'每列的数据类型:\n{df.dtypes}')
```

    前几行数据：
       carat      cut color clarity  depth  table  price     x     y     z
    1   0.23    Ideal     E     SI2   61.5   55.0    326  3.95  3.98  2.43
    2   0.21  Premium     E     SI1   59.8   61.0    326  3.89  3.84  2.31
    3   0.23     Good     E     VS1   56.9   65.0    327  4.05  4.07  2.31
    4   0.29  Premium     I     VS2   62.4   58.0    334  4.20  4.23  2.63
    5   0.31     Good     J     SI2   63.3   58.0    335  4.34  4.35  2.75
    数据大小：
    (53940, 10)
    每列的数据类型:
    carat      float64
    cut         object
    color       object
    clarity     object
    depth      float64
    table      float64
    price        int64
    x          float64
    y          float64
    z          float64
    dtype: object
    


```python
df2 = pd.get_dummies(data=df,columns=['cut'])
from sklearn.preprocessing import LabelEncoder
le_color = LabelEncoder()
le_color.fit(df['color'].unique())
df2['color'] = le_color.transform(df['color'])

le_clarity = LabelEncoder()
le_clarity.fit(df['clarity'].unique())
df2['clarity'] = le_clarity.transform(df['clarity'])
df2.head()
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
      <th>carat</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut_Fair</th>
      <th>cut_Good</th>
      <th>cut_Ideal</th>
      <th>cut_Premium</th>
      <th>cut_Very Good</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.23</td>
      <td>1</td>
      <td>3</td>
      <td>61.5</td>
      <td>55.0</td>
      <td>326</td>
      <td>3.95</td>
      <td>3.98</td>
      <td>2.43</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.21</td>
      <td>1</td>
      <td>2</td>
      <td>59.8</td>
      <td>61.0</td>
      <td>326</td>
      <td>3.89</td>
      <td>3.84</td>
      <td>2.31</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.23</td>
      <td>1</td>
      <td>4</td>
      <td>56.9</td>
      <td>65.0</td>
      <td>327</td>
      <td>4.05</td>
      <td>4.07</td>
      <td>2.31</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.29</td>
      <td>5</td>
      <td>5</td>
      <td>62.4</td>
      <td>58.0</td>
      <td>334</td>
      <td>4.20</td>
      <td>4.23</td>
      <td>2.63</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.31</td>
      <td>6</td>
      <td>3</td>
      <td>63.3</td>
      <td>58.0</td>
      <td>335</td>
      <td>4.34</td>
      <td>4.35</td>
      <td>2.75</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# 二、简单的数据分析


```python
with pd.option_context('display.max_columns',None):
    print(df2.describe())
```

                  carat         color       clarity         depth         table  \
    count  53940.000000  53940.000000  53940.000000  53940.000000  53940.000000   
    mean       0.797940      2.594197      3.835150     61.749405     57.457184   
    std        0.474011      1.701105      1.724591      1.432621      2.234491   
    min        0.200000      0.000000      0.000000     43.000000     43.000000   
    25%        0.400000      1.000000      2.000000     61.000000     56.000000   
    50%        0.700000      3.000000      4.000000     61.800000     57.000000   
    75%        1.040000      4.000000      5.000000     62.500000     59.000000   
    max        5.010000      6.000000      7.000000     79.000000     95.000000   
    
                  price             x             y             z      cut_Fair  \
    count  53940.000000  53940.000000  53940.000000  53940.000000  53940.000000   
    mean    3932.799722      5.731157      5.734526      3.538734      0.029848   
    std     3989.439738      1.121761      1.142135      0.705699      0.170169   
    min      326.000000      0.000000      0.000000      0.000000      0.000000   
    25%      950.000000      4.710000      4.720000      2.910000      0.000000   
    50%     2401.000000      5.700000      5.710000      3.530000      0.000000   
    75%     5324.250000      6.540000      6.540000      4.040000      0.000000   
    max    18823.000000     10.740000     58.900000     31.800000      1.000000   
    
               cut_Good     cut_Ideal   cut_Premium  cut_Very Good  
    count  53940.000000  53940.000000  53940.000000   53940.000000  
    mean       0.090953      0.399537      0.255673       0.223990  
    std        0.287545      0.489808      0.436243       0.416919  
    min        0.000000      0.000000      0.000000       0.000000  
    25%        0.000000      0.000000      0.000000       0.000000  
    50%        0.000000      0.000000      0.000000       0.000000  
    75%        0.000000      1.000000      1.000000       0.000000  
    max        1.000000      1.000000      1.000000       1.000000  
    

# 三、数据清洗


```python
df2[df2.isnull().sum(axis=1)<=3]
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
      <th>carat</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut_Fair</th>
      <th>cut_Good</th>
      <th>cut_Ideal</th>
      <th>cut_Premium</th>
      <th>cut_Very Good</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.23</td>
      <td>1</td>
      <td>3</td>
      <td>61.5</td>
      <td>55.0</td>
      <td>326</td>
      <td>3.95</td>
      <td>3.98</td>
      <td>2.43</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.21</td>
      <td>1</td>
      <td>2</td>
      <td>59.8</td>
      <td>61.0</td>
      <td>326</td>
      <td>3.89</td>
      <td>3.84</td>
      <td>2.31</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.23</td>
      <td>1</td>
      <td>4</td>
      <td>56.9</td>
      <td>65.0</td>
      <td>327</td>
      <td>4.05</td>
      <td>4.07</td>
      <td>2.31</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.29</td>
      <td>5</td>
      <td>5</td>
      <td>62.4</td>
      <td>58.0</td>
      <td>334</td>
      <td>4.20</td>
      <td>4.23</td>
      <td>2.63</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.31</td>
      <td>6</td>
      <td>3</td>
      <td>63.3</td>
      <td>58.0</td>
      <td>335</td>
      <td>4.34</td>
      <td>4.35</td>
      <td>2.75</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.24</td>
      <td>6</td>
      <td>7</td>
      <td>62.8</td>
      <td>57.0</td>
      <td>336</td>
      <td>3.94</td>
      <td>3.96</td>
      <td>2.48</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.24</td>
      <td>5</td>
      <td>6</td>
      <td>62.3</td>
      <td>57.0</td>
      <td>336</td>
      <td>3.95</td>
      <td>3.98</td>
      <td>2.47</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.26</td>
      <td>4</td>
      <td>2</td>
      <td>61.9</td>
      <td>55.0</td>
      <td>337</td>
      <td>4.07</td>
      <td>4.11</td>
      <td>2.53</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.22</td>
      <td>1</td>
      <td>5</td>
      <td>65.1</td>
      <td>61.0</td>
      <td>337</td>
      <td>3.87</td>
      <td>3.78</td>
      <td>2.49</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.23</td>
      <td>4</td>
      <td>4</td>
      <td>59.4</td>
      <td>61.0</td>
      <td>338</td>
      <td>4.00</td>
      <td>4.05</td>
      <td>2.39</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.30</td>
      <td>6</td>
      <td>2</td>
      <td>64.0</td>
      <td>55.0</td>
      <td>339</td>
      <td>4.25</td>
      <td>4.28</td>
      <td>2.73</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.23</td>
      <td>6</td>
      <td>4</td>
      <td>62.8</td>
      <td>56.0</td>
      <td>340</td>
      <td>3.93</td>
      <td>3.90</td>
      <td>2.46</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.22</td>
      <td>2</td>
      <td>2</td>
      <td>60.4</td>
      <td>61.0</td>
      <td>342</td>
      <td>3.88</td>
      <td>3.84</td>
      <td>2.33</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.31</td>
      <td>6</td>
      <td>3</td>
      <td>62.2</td>
      <td>54.0</td>
      <td>344</td>
      <td>4.35</td>
      <td>4.37</td>
      <td>2.71</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.20</td>
      <td>1</td>
      <td>3</td>
      <td>60.2</td>
      <td>62.0</td>
      <td>345</td>
      <td>3.79</td>
      <td>3.75</td>
      <td>2.27</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.32</td>
      <td>1</td>
      <td>0</td>
      <td>60.9</td>
      <td>58.0</td>
      <td>345</td>
      <td>4.38</td>
      <td>4.42</td>
      <td>2.68</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.30</td>
      <td>5</td>
      <td>3</td>
      <td>62.0</td>
      <td>54.0</td>
      <td>348</td>
      <td>4.31</td>
      <td>4.34</td>
      <td>2.68</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.30</td>
      <td>6</td>
      <td>2</td>
      <td>63.4</td>
      <td>54.0</td>
      <td>351</td>
      <td>4.23</td>
      <td>4.29</td>
      <td>2.70</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.30</td>
      <td>6</td>
      <td>2</td>
      <td>63.8</td>
      <td>56.0</td>
      <td>351</td>
      <td>4.23</td>
      <td>4.26</td>
      <td>2.71</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.30</td>
      <td>6</td>
      <td>2</td>
      <td>62.7</td>
      <td>59.0</td>
      <td>351</td>
      <td>4.21</td>
      <td>4.27</td>
      <td>2.66</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.30</td>
      <td>5</td>
      <td>3</td>
      <td>63.3</td>
      <td>56.0</td>
      <td>351</td>
      <td>4.26</td>
      <td>4.30</td>
      <td>2.71</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.23</td>
      <td>1</td>
      <td>5</td>
      <td>63.8</td>
      <td>55.0</td>
      <td>352</td>
      <td>3.85</td>
      <td>3.92</td>
      <td>2.48</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.23</td>
      <td>4</td>
      <td>4</td>
      <td>61.0</td>
      <td>57.0</td>
      <td>353</td>
      <td>3.94</td>
      <td>3.96</td>
      <td>2.41</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.31</td>
      <td>6</td>
      <td>2</td>
      <td>59.4</td>
      <td>62.0</td>
      <td>353</td>
      <td>4.39</td>
      <td>4.43</td>
      <td>2.62</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.31</td>
      <td>6</td>
      <td>2</td>
      <td>58.1</td>
      <td>62.0</td>
      <td>353</td>
      <td>4.44</td>
      <td>4.47</td>
      <td>2.59</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.23</td>
      <td>3</td>
      <td>7</td>
      <td>60.4</td>
      <td>58.0</td>
      <td>354</td>
      <td>3.97</td>
      <td>4.01</td>
      <td>2.41</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.24</td>
      <td>5</td>
      <td>4</td>
      <td>62.5</td>
      <td>57.0</td>
      <td>355</td>
      <td>3.97</td>
      <td>3.94</td>
      <td>2.47</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.30</td>
      <td>6</td>
      <td>5</td>
      <td>62.2</td>
      <td>57.0</td>
      <td>357</td>
      <td>4.28</td>
      <td>4.30</td>
      <td>2.67</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.23</td>
      <td>0</td>
      <td>5</td>
      <td>60.5</td>
      <td>61.0</td>
      <td>357</td>
      <td>3.96</td>
      <td>3.97</td>
      <td>2.40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.23</td>
      <td>2</td>
      <td>4</td>
      <td>60.9</td>
      <td>57.0</td>
      <td>357</td>
      <td>3.96</td>
      <td>3.99</td>
      <td>2.42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>53911</th>
      <td>0.70</td>
      <td>1</td>
      <td>2</td>
      <td>60.5</td>
      <td>58.0</td>
      <td>2753</td>
      <td>5.74</td>
      <td>5.77</td>
      <td>3.48</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53912</th>
      <td>0.57</td>
      <td>1</td>
      <td>1</td>
      <td>59.8</td>
      <td>60.0</td>
      <td>2753</td>
      <td>5.43</td>
      <td>5.38</td>
      <td>3.23</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53913</th>
      <td>0.61</td>
      <td>2</td>
      <td>6</td>
      <td>61.8</td>
      <td>59.0</td>
      <td>2753</td>
      <td>5.48</td>
      <td>5.40</td>
      <td>3.36</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53914</th>
      <td>0.80</td>
      <td>3</td>
      <td>5</td>
      <td>64.2</td>
      <td>58.0</td>
      <td>2753</td>
      <td>5.84</td>
      <td>5.81</td>
      <td>3.74</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53915</th>
      <td>0.84</td>
      <td>5</td>
      <td>4</td>
      <td>63.7</td>
      <td>59.0</td>
      <td>2753</td>
      <td>5.94</td>
      <td>5.90</td>
      <td>3.77</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53916</th>
      <td>0.77</td>
      <td>1</td>
      <td>3</td>
      <td>62.1</td>
      <td>56.0</td>
      <td>2753</td>
      <td>5.84</td>
      <td>5.86</td>
      <td>3.63</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53917</th>
      <td>0.74</td>
      <td>0</td>
      <td>2</td>
      <td>63.1</td>
      <td>59.0</td>
      <td>2753</td>
      <td>5.71</td>
      <td>5.74</td>
      <td>3.61</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53918</th>
      <td>0.90</td>
      <td>6</td>
      <td>2</td>
      <td>63.2</td>
      <td>60.0</td>
      <td>2753</td>
      <td>6.12</td>
      <td>6.09</td>
      <td>3.86</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53919</th>
      <td>0.76</td>
      <td>5</td>
      <td>4</td>
      <td>59.3</td>
      <td>62.0</td>
      <td>2753</td>
      <td>5.93</td>
      <td>5.85</td>
      <td>3.49</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53920</th>
      <td>0.76</td>
      <td>5</td>
      <td>6</td>
      <td>62.2</td>
      <td>55.0</td>
      <td>2753</td>
      <td>5.89</td>
      <td>5.87</td>
      <td>3.66</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53921</th>
      <td>0.70</td>
      <td>1</td>
      <td>5</td>
      <td>62.4</td>
      <td>60.0</td>
      <td>2755</td>
      <td>5.57</td>
      <td>5.61</td>
      <td>3.49</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53922</th>
      <td>0.70</td>
      <td>1</td>
      <td>5</td>
      <td>62.8</td>
      <td>60.0</td>
      <td>2755</td>
      <td>5.59</td>
      <td>5.65</td>
      <td>3.53</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53923</th>
      <td>0.70</td>
      <td>0</td>
      <td>4</td>
      <td>63.1</td>
      <td>59.0</td>
      <td>2755</td>
      <td>5.67</td>
      <td>5.58</td>
      <td>3.55</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53924</th>
      <td>0.73</td>
      <td>5</td>
      <td>5</td>
      <td>61.3</td>
      <td>56.0</td>
      <td>2756</td>
      <td>5.80</td>
      <td>5.84</td>
      <td>3.57</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53925</th>
      <td>0.73</td>
      <td>5</td>
      <td>5</td>
      <td>61.6</td>
      <td>55.0</td>
      <td>2756</td>
      <td>5.82</td>
      <td>5.84</td>
      <td>3.59</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53926</th>
      <td>0.79</td>
      <td>5</td>
      <td>2</td>
      <td>61.6</td>
      <td>56.0</td>
      <td>2756</td>
      <td>5.95</td>
      <td>5.97</td>
      <td>3.67</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53927</th>
      <td>0.71</td>
      <td>1</td>
      <td>2</td>
      <td>61.9</td>
      <td>56.0</td>
      <td>2756</td>
      <td>5.71</td>
      <td>5.73</td>
      <td>3.54</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53928</th>
      <td>0.79</td>
      <td>2</td>
      <td>2</td>
      <td>58.1</td>
      <td>59.0</td>
      <td>2756</td>
      <td>6.06</td>
      <td>6.13</td>
      <td>3.54</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53929</th>
      <td>0.79</td>
      <td>1</td>
      <td>3</td>
      <td>61.4</td>
      <td>58.0</td>
      <td>2756</td>
      <td>6.03</td>
      <td>5.96</td>
      <td>3.68</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53930</th>
      <td>0.71</td>
      <td>3</td>
      <td>4</td>
      <td>61.4</td>
      <td>56.0</td>
      <td>2756</td>
      <td>5.76</td>
      <td>5.73</td>
      <td>3.53</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53931</th>
      <td>0.71</td>
      <td>1</td>
      <td>2</td>
      <td>60.5</td>
      <td>55.0</td>
      <td>2756</td>
      <td>5.79</td>
      <td>5.74</td>
      <td>3.49</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53932</th>
      <td>0.71</td>
      <td>2</td>
      <td>2</td>
      <td>59.8</td>
      <td>62.0</td>
      <td>2756</td>
      <td>5.74</td>
      <td>5.73</td>
      <td>3.43</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53933</th>
      <td>0.70</td>
      <td>1</td>
      <td>5</td>
      <td>60.5</td>
      <td>59.0</td>
      <td>2757</td>
      <td>5.71</td>
      <td>5.76</td>
      <td>3.47</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53934</th>
      <td>0.70</td>
      <td>1</td>
      <td>5</td>
      <td>61.2</td>
      <td>59.0</td>
      <td>2757</td>
      <td>5.69</td>
      <td>5.72</td>
      <td>3.49</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53935</th>
      <td>0.72</td>
      <td>0</td>
      <td>2</td>
      <td>62.7</td>
      <td>59.0</td>
      <td>2757</td>
      <td>5.69</td>
      <td>5.73</td>
      <td>3.58</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53936</th>
      <td>0.72</td>
      <td>0</td>
      <td>2</td>
      <td>60.8</td>
      <td>57.0</td>
      <td>2757</td>
      <td>5.75</td>
      <td>5.76</td>
      <td>3.50</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53937</th>
      <td>0.72</td>
      <td>0</td>
      <td>2</td>
      <td>63.1</td>
      <td>55.0</td>
      <td>2757</td>
      <td>5.69</td>
      <td>5.75</td>
      <td>3.61</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53938</th>
      <td>0.70</td>
      <td>0</td>
      <td>2</td>
      <td>62.8</td>
      <td>60.0</td>
      <td>2757</td>
      <td>5.66</td>
      <td>5.68</td>
      <td>3.56</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53939</th>
      <td>0.86</td>
      <td>4</td>
      <td>3</td>
      <td>61.0</td>
      <td>58.0</td>
      <td>2757</td>
      <td>6.15</td>
      <td>6.12</td>
      <td>3.74</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53940</th>
      <td>0.75</td>
      <td>0</td>
      <td>3</td>
      <td>62.2</td>
      <td>55.0</td>
      <td>2757</td>
      <td>5.83</td>
      <td>5.87</td>
      <td>3.64</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>53940 rows × 14 columns</p>
</div>



# 四、数据写成矩阵形式并划分数据集


```python
X = df2.drop('price',axis=1).values
y = df2['price'].values # Y是 1x53940大小的，还没调成53940x1，问题应该不大
print(f'自变量矩阵形状：{X.shape}')
print(f'因变量矩阵形状：{y.shape}')
print(f'X[0] = {X[0]}') # 打印第0行看一下
print(f'y[0] = {y[0]}')
print(y)
```

    自变量矩阵形状：(53940, 13)
    因变量矩阵形状：(53940,)
    X[0] = [ 0.23  1.    3.   61.5  55.    3.95  3.98  2.43  0.    0.    1.    0.
      0.  ]
    y[0] = 326
    [ 326  326  327 ... 2757 2757 2757]
    


```python
from sklearn.model_selection import ShuffleSplit
seed = 3
rs = ShuffleSplit(n_splits=1,test_size=0.2,random_state=seed)
tv_idx,test_idx = next(rs.split(X))
print(f't+v 长度：{len(tv_idx)}, test 长度：{len(test_idx)}')
tvX,tvy = X[tv_idx],y[tv_idx]
testX,testy = X[test_idx],y[test_idx]
```

    t+v 长度：43152, test 长度：10788
    

# 五、训练模型


```python
rs2 = ShuffleSplit(n_splits=1,test_size=0.3,random_state=3)
train_idx,val_idx = next(rs2.split(tvX))
tX,ty = tvX[train_idx],tvy[train_idx]
vX,vy = tvX[val_idx],tvy[val_idx]
```


```python
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(max_depth=3)
tree_reg.fit(tX,ty)
```




    DecisionTreeRegressor(criterion='mse', max_depth=3, max_features=None,
               max_leaf_nodes=None, min_impurity_decrease=0.0,
               min_impurity_split=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               presort=False, random_state=None, splitter='best')




```python
# 把树的样子画出来
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(25,20))

from sklearn.tree import DecisionTreeClassifier, plot_tree

_ = plot_tree(tree_reg,feature_names=list(df2.columns),filled=True)
text_representation = tee.export_text(tree_reg)
print(text_representation)
```


    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    <ipython-input-10-852d8b2fef4b> in <module>()
          3 fig = plt.figure(figsize=(25,20))
          4 
    ----> 5 from sklearn.tree import DecisionTreeClassifier, plot_tree
          6 
          7 _ = plot_tree(tree_reg,feature_names=list(df2.columns),filled=True)
    

    ImportError: cannot import name 'plot_tree'



    <Figure size 1800x1440 with 0 Axes>



```python
import sklearn
print("Sklearn verion is {}".format(sklearn.__version__))
```

    Sklearn verion is 0.19.1
    

sklearn版本低于2.0，没有plot_tree，升级sklearn
https://blog.csdn.net/qq_37741588/article/details/95079217

# 重复实验——交叉检验


```python
import time
start_time = time.process_time()
from sklearn.model_selection import GridSearchCV
param = {
    'criterion':['mse'],
    'max_depth':[5,10,20,30],
    'min_samples_split':[20,40,80,160],
    'min_samples_leaf':[10,20,40,80]}
grid = GridSearchCV(DecisionTreeRegressor(),param_grid=param,cv=5)
grid.fit(tvX,tvy)
time_spent = time.process_time()-start_time
print(f'最优分类器:{grid.best_params_}')
print(f'最优分数:{grid.best_score_}')
print(f'时间：{time_spent:.2f} 秒')  
```

    最优分类器:{'criterion': 'mse', 'max_depth': 30, 'min_samples_leaf': 10, 'min_samples_split': 20}
    最优分数:0.9740593570746284
    时间：27.98 秒
    

用最佳的参数训练


```python
dt = DecisionTreeRegressor(**grid.best_params_)
dt.fit(tvX,tvy)
test_pred = dt.predict(testX)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(testy,test_pred)
print(f'测试集上 mse: {mse}')
import numpy as np
basey = [np.mean(tvy)]*len(testy)
baseMSE = mean_squared_error(testy,basey)
print(f'测试集上 nmse: {mse/baseMSE}')
```

    测试集上 mse: 412297.5609538985
    测试集上 nmse: 0.026284517935997623
    


```python
plt.scatter(testy,test_pred)
plt.xlabel('真实价格',fontproperties = 'SimHei')
plt.ylabel('预测价格',fontproperties = 'SimHei')
plt.title('预测价格 vs 真实价格',fontproperties = 'SimHei')
plt.show()
```


![output_20_0](C:/Users/Tiny/Documents/GitHub/xieyi11.github.io/_posts/imgs/output_20_0.png)


# 随机森林


```python
import time
start_time = time.process_time()
from sklearn.ensemble import RandomForestRegressor
```


    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    <ipython-input-46-027608dbdbc0> in <module>()
          1 import time
          2 start_time = time.process_time()
    ----> 3 from sklearn.ensemble import RandomForestRegressor
    

    D:\anaconda\lib\site-packages\sklearn\ensemble\__init__.py in <module>()
          5 
          6 from ._base import BaseEnsemble
    ----> 7 from ._forest import RandomForestClassifier
          8 from ._forest import RandomForestRegressor
          9 from ._forest import RandomTreesEmbedding
    

    D:\anaconda\lib\site-packages\sklearn\ensemble\_forest.py in <module>()
         51 from joblib import Parallel, delayed
         52 
    ---> 53 from ..base import ClassifierMixin, RegressorMixin, MultiOutputMixin
         54 from ..metrics import r2_score
         55 from ..preprocessing import OneHotEncoder
    

    ImportError: cannot import name 'MultiOutputMixin'



```python
param = {
    'n_estimators':[50,100],
    'max_depth':[10,20,30],
    'min_samples_split':[20],
    'min_samples_leaf':[5,10]}
grid = GridSearchCV(RandomForestRegressor(),param_grid=param,cv=3)
grid.fit(tvX,tvy)
print(f'CPU 时间：{time.process_time()-start_time}')
print(f'最优分类器:{grid.best_params_}')
print(f'最优分数:{grid.best_score_}')
```


    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    <ipython-input-30-5ca4936501d1> in <module>()
          1 import time
          2 start_time = time.process_time()
    ----> 3 from sklearn.ensemble import RandomForestRegressor
          4 param = {
          5     'n_estimators':[50,100],
    

    D:\anaconda\lib\site-packages\sklearn\ensemble\__init__.py in <module>()
          5 
          6 from ._base import BaseEnsemble
    ----> 7 from ._forest import RandomForestClassifier
          8 from ._forest import RandomForestRegressor
          9 from ._forest import RandomTreesEmbedding
    

    D:\anaconda\lib\site-packages\sklearn\ensemble\_forest.py in <module>()
         51 from joblib import Parallel, delayed
         52 
    ---> 53 from ..base import ClassifierMixin, RegressorMixin, MultiOutputMixin
         54 from ..metrics import r2_score
         55 from ..preprocessing import OneHotEncoder
    

    ImportError: cannot import name 'MultiOutputMixin'

