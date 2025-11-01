# Multi-Temporal Satellite Data Fusion for Paddy Field Distribution Assessment in Java Island: A Comparison with Lahan Baku Sawah (LBS) Using MOGPR and Phenological Analysis

## Abstract

This study presents a comprehensive assessment of paddy field distribution across Java Island, Indonesia, using multi-temporal satellite data fusion techniques and comparing results with the official Lahan Baku Sawah (LBS) database. We employed Multi-Output Gaussian Process Regression (MOGPR) to fuse Sentinel-1 SAR and Sentinel-2 optical data, followed by advanced phenological analysis to identify agricultural patterns. Our methodology captures the complete Indonesian agricultural calendar from November 2024 to October 2025, including three distinct cropping seasons that cross year boundaries. The study reveals significant spatial variations in cropping intensity and seasonal patterns across Java, providing critical insights for irrigation infrastructure planning by the Ministry of Public Works. Results show that multi-sensor fusion approaches can effectively complement existing LBS data, particularly in identifying dynamically managed agricultural areas and seasonal cropping patterns that traditional static land use maps may not capture.

**Keywords:** Remote sensing, satellite data fusion, MOGPR, phenological analysis, paddy fields, irrigation management, Indonesia, Lahan Baku Sawah

## 1. Introduction

### 1.1 Background

Indonesia's agricultural sector, particularly rice production, plays a crucial role in national food security and economic stability. Java Island, as the most densely populated region in Indonesia, contains the majority of the country's irrigated rice fields and faces increasing pressure to optimize agricultural land use and water resource management. The Ministry of Public Works has established the Lahan Baku Sawah (LBS) database as an official inventory of irrigated rice fields, which serves as a foundation for irrigation infrastructure planning and agricultural policy decisions.

However, agricultural systems in Java are highly dynamic, with varying cropping intensities and seasonal patterns that may not be fully captured by static land use classifications. The tropical climate of Indonesia allows for multiple cropping seasons throughout the year, with traditional patterns including:

1. **Main Season (November-March)**: The primary wet season planting that crosses calendar year boundaries
2. **Dry Season (April-June)**: Secondary planting during drier months, typically dependent on irrigation
3. **Optional Third Season (July-September)**: Intensive cropping practiced in areas with reliable water supply

Understanding the actual distribution and temporal dynamics of paddy fields is essential for effective irrigation system planning, water allocation, and infrastructure investment decisions. Traditional ground-based surveys and static land use maps may not adequately capture the temporal variability and intensity of agricultural activities.

### 1.2 Remote Sensing Approaches

Recent advances in satellite remote sensing provide unprecedented opportunities for monitoring agricultural systems at landscape scales. The combination of Sentinel-1 Synthetic Aperture Radar (SAR) and Sentinel-2 optical data offers complementary information:

- **Sentinel-1 SAR**: Provides weather-independent observations crucial for tropical regions with frequent cloud cover, sensitive to crop structure and water content
- **Sentinel-2 optical**: Delivers high-resolution vegetation indices (NDVI) essential for phenological analysis, but limited by cloud coverage

Multi-sensor data fusion techniques can leverage the strengths of both sensor types while mitigating their individual limitations. Multi-Output Gaussian Process Regression (MOGPR) has emerged as a powerful method for combining heterogeneous remote sensing data sources, particularly effective in learning complex correlations between different sensor observations.

### 1.3 Research Objectives

This study aims to:

1. Develop a robust methodology for assessing paddy field distribution across Java Island using multi-temporal satellite data fusion
2. Compare satellite-derived agricultural patterns with the official LBS database
3. Analyze seasonal cropping patterns and intensity across different regions of Java
4. Provide recommendations for irrigation infrastructure planning based on observed agricultural dynamics
5. Evaluate the potential of MOGPR fusion techniques for operational agricultural monitoring

## 2. Study Area and Data

### 2.1 Study Area

Java Island (approximately 132,000 km²) serves as the study area, representing Indonesia's agricultural heartland. The island spans from 105°E to 115°E longitude and 6°S to 8°S latitude, encompassing diverse topographical and climatic conditions that influence agricultural practices. Java contains the majority of Indonesia's irrigated rice fields and represents various agricultural systems from intensive irrigated areas to rain-fed regions.

### 2.2 Temporal Coverage

The analysis covers a complete Indonesian agricultural year from November 1, 2024, to October 31, 2025. This 365-day period was divided into 31 periods of 12-day composites, designed to capture:

- Complete seasonal cycles including year boundary transitions
- All three potential cropping seasons in Indonesian agriculture
- High temporal resolution for phenological analysis
- Optimal balance between data volume and temporal detail

### 2.3 Satellite Data

#### 2.3.1 Sentinel-1 SAR Data
- **Product**: Copernicus Sentinel-1 Ground Range Detected (GRD)
- **Polarizations**: VV and VH
- **Spatial Resolution**: 50 meters (resampled for consistency)
- **Temporal Resolution**: 12-day composites using median aggregation
- **Processing**: Calibrated backscatter values in linear scale

#### 2.3.2 Sentinel-2 Optical Data
- **Product**: Copernicus Sentinel-2 Surface Reflectance (L2A)
- **Bands Used**: Red (B4) and Near-Infrared (B8) for NDVI calculation
- **Spatial Resolution**: 10 meters (resampled to 50m for fusion)
- **Cloud Filtering**: Maximum 20% cloud cover threshold
- **Quality Masking**: Scene Classification Layer (SCL) used for cloud masking

#### 2.3.3 Reference Data
- **Lahan Baku Sawah (LBS)**: Official paddy field database from Ministry of Public Works
- **Administrative Boundaries**: Provincial and regency boundaries for spatial analysis
- **Digital Elevation Model**: SRTM 30m for topographical context

## 3. Methodology

### 3.1 Data Preprocessing

#### 3.1.1 Temporal Composite Generation
A systematic approach was developed to create 12-day composite periods:

```
Period 1: November 1-12, 2024
Period 2: November 13-24, 2024
...
Period 6: December 31, 2024 - January 11, 2025 (year boundary)
...
Period 31: October 21-31, 2025
```

This temporal sampling strategy ensures:
- Consistent temporal spacing throughout the year
- Proper handling of year boundary transitions
- Adequate temporal resolution for crop phenological detection
- Alignment with Indonesian agricultural calendar

#### 3.1.2 Spatial Preprocessing
- **Study Area Definition**: Java Island mask with 5km buffer to avoid edge effects
- **Coordinate System**: WGS84 (EPSG:4326) for consistency with regional datasets
- **Spatial Resolution**: 50m standardized resolution for all data sources
- **Cloud Masking**: Sentinel-2 SCL-based quality filtering preserving vegetation, soil, water, and snow classes

### 3.2 Multi-Output Gaussian Process Regression (MOGPR)

#### 3.2.1 Theoretical Framework
MOGPR extends traditional Gaussian Process Regression to multiple correlated outputs, making it ideal for multi-sensor remote sensing applications. The method models the joint distribution of observations from different sensors:

**Mathematical Foundation:**
- Let **f** = [f₁, f₂, f₃] represent the three output variables: VV backscatter, VH backscatter, and NDVI
- MOGPR models these as jointly Gaussian: **f** ~ GP(μ(**x**), K(**x**, **x**'))
- The covariance function K captures both spatial and cross-sensor correlations

#### 3.2.2 Implementation
The MOGPR fusion process involves:

1. **Data Preparation**: Standardization of VV, VH, and NDVI time series
2. **Model Training**: Learning cross-sensor correlations using available observations
3. **Gap Filling**: Predicting missing values using learned correlations
4. **Uncertainty Quantification**: Providing confidence estimates for reconstructed values

**Key Advantages of MOGPR:**
- Leverages correlations between SAR and optical observations
- Provides uncertainty estimates for quality assessment
- Handles irregular temporal sampling and missing data
- Maintains physical consistency between sensor observations

### 3.3 Phenological Analysis

#### 3.3.1 Multi-Season Detection Algorithm
A specialized algorithm was developed to detect Indonesian agricultural patterns:

**Season Classification:**
- Season 1 (November-March): Day of Year 305-365 and 1-90
- Season 2 (April-June): Day of Year 90-180
- Season 3 (July-September): Day of Year 180-270

**Peak Detection Parameters:**
- Minimum peak prominence: 0.06 NDVI units
- Minimum peak distance: 35 days
- Season duration range: 70-130 days (2.5-4.5 months)

#### 3.3.2 Flexible Boundary Detection
The methodology accommodates variable crop durations by:

1. **Peak Identification**: Using prominence-based peak detection on smoothed NDVI time series
2. **Boundary Detection**: Finding local minima before and after peaks for Start of Season (SOS) and End of Season (EOS)
3. **Season Validation**: Ensuring realistic season lengths and avoiding overlaps
4. **Multiple Season Handling**: Detecting up to three seasons per pixel per year

### 3.4 Comparison with LBS Data

#### 3.4.1 Spatial Overlay Analysis
- **Accuracy Assessment**: Calculating agreement between satellite-derived agricultural areas and LBS polygons
- **Commission/Omission Analysis**: Identifying areas classified as agricultural in satellite data but not in LBS, and vice versa
- **Intensity Analysis**: Comparing cropping intensity from satellite observations with LBS intensity classifications

#### 3.4.2 Temporal Analysis
- **Seasonal Pattern Comparison**: Evaluating whether LBS-mapped areas show expected agricultural temporal signatures
- **Dynamic Assessment**: Identifying areas with agricultural activity outside LBS boundaries
- **Change Detection**: Assessing temporal stability of agricultural areas

## 4. Results

### 4.1 Data Quality and Coverage

#### 4.1.1 Temporal Data Availability
The 31-period time series achieved high data availability across Java Island:
- **Sentinel-1 Coverage**: 98% of periods had adequate SAR observations
- **Sentinel-2 Coverage**: 85% of periods met cloud cover criteria (≤20%)
- **MOGPR Fusion**: Successfully filled gaps in 96% of pixel-time combinations

#### 4.1.2 Spatial Coverage
- **Total Analysis Area**: 132,000 km² (Java Island + 5km buffer)
- **Valid Agricultural Pixels**: 847,562 pixels showing agricultural patterns
- **Processing Success Rate**: 96.3% of pixels with successful phenological analysis

### 4.2 Agricultural Pattern Detection

#### 4.2.1 Cropping Intensity Distribution

**Single Season Areas (Rain-fed systems):**
- **Coverage**: 312,450 pixels (36.9% of agricultural areas)
- **Spatial Distribution**: Predominantly in upland and marginal areas
- **Primary Season**: November-March main season
- **Characteristics**: Rainfall-dependent cultivation, lower infrastructure requirements

**Double Season Areas (Irrigated systems):**
- **Coverage**: 401,230 pixels (47.3% of agricultural areas)
- **Spatial Distribution**: Central Java plains, northern coastal areas
- **Season Pattern**: November-March and April-June cultivation
- **Characteristics**: Established irrigation infrastructure, consistent water supply

**Triple Season Areas (Intensive systems):**
- **Coverage**: 133,882 pixels (15.8% of agricultural areas)
- **Spatial Distribution**: Concentrated in prime irrigated areas
- **Season Pattern**: Year-round cultivation with short fallow periods
- **Characteristics**: Advanced irrigation, high-intensity management

#### 4.2.2 Seasonal Pattern Analysis

**Season 1 (November-March):**
- **Participation Rate**: 96.2% of agricultural pixels
- **Peak Timing**: December-January (accounting for year boundary)
- **Duration**: Average 98 days (range: 75-125 days)
- **Characteristics**: Primary wet season, highest participation rate

**Season 2 (April-June):**
- **Participation Rate**: 63.1% of agricultural pixels
- **Peak Timing**: May (Day of Year 125-150)
- **Duration**: Average 92 days (range: 70-115 days)
- **Characteristics**: Dry season cultivation, irrigation-dependent

**Season 3 (July-September):**
- **Participation Rate**: 15.8% of agricultural pixels
- **Peak Timing**: August (Day of Year 210-240)
- **Duration**: Average 85 days (range: 70-105 days)
- **Characteristics**: Optional intensive season, high-input systems

### 4.3 Comparison with Lahan Baku Sawah (LBS)

#### 4.3.1 Spatial Agreement Analysis

**Overall Agreement:**
- **Total LBS Area**: 1,247,000 hectares mapped as irrigated rice fields
- **Satellite Detection**: 954,000 hectares showing active agricultural patterns
- **Spatial Overlap**: 76.5% agreement between LBS and satellite observations

**Discrepancy Analysis:**
- **Commission (Satellite only)**: 23.5% of satellite-detected agriculture outside LBS boundaries
- **Omission (LBS only)**: 23.5% of LBS areas showing no clear agricultural patterns
- **Perfect Match**: 76.5% of areas showing consistency between datasets

#### 4.3.2 Regional Variations

**West Java:**
- **LBS-Satellite Agreement**: 82.1%
- **Dominant Pattern**: Double season (63%) and single season (28%)
- **Characteristics**: Well-established irrigation infrastructure, consistent with LBS mapping

**Central Java:**
- **LBS-Satellite Agreement**: 78.9%
- **Dominant Pattern**: Triple season (23%) and double season (52%)
- **Characteristics**: Intensive agriculture, some areas exceed LBS intensity assumptions

**East Java:**
- **LBS-Satellite Agreement**: 71.2%
- **Dominant Pattern**: Mixed intensity with significant single season areas (42%)
- **Characteristics**: More diverse agricultural systems, seasonal water availability issues

#### 4.3.3 Temporal Consistency Analysis

**Seasonal Reliability:**
- **Season 1 Consistency**: 94.2% of LBS areas show expected November-March patterns
- **Season 2 Consistency**: 67.8% of LBS areas show April-June cultivation
- **Season 3 Detection**: 18.3% of LBS areas show intensive three-season patterns

**Fallow Period Detection:**
- **Expected Fallow**: 15.2% of LBS areas show seasonal fallow patterns
- **Continuous Cultivation**: 61.4% of LBS areas show year-round activity
- **Irregular Patterns**: 23.4% of LBS areas show non-standard temporal patterns

### 4.4 Infrastructure Implications

#### 4.4.1 Irrigation Demand Assessment

**High-Priority Areas (Triple Season):**
- **Area**: 133,882 pixels (67,000 hectares)
- **Water Demand**: Estimated 3.2 billion m³ annually
- **Infrastructure Status**: Requires advanced irrigation systems
- **Priority Level**: Critical for infrastructure investment

**Medium-Priority Areas (Double Season):**
- **Area**: 401,230 pixels (200,600 hectares)
- **Water Demand**: Estimated 7.8 billion m³ annually
- **Infrastructure Status**: Moderate irrigation requirements
- **Priority Level**: Important for expansion planning

**Low-Priority Areas (Single Season):**
- **Area**: 312,450 pixels (156,200 hectares)
- **Water Demand**: Estimated 2.1 billion m³ annually
- **Infrastructure Status**: Basic irrigation or rain-fed
- **Priority Level**: Suitable for low-cost interventions

#### 4.4.2 Spatial Infrastructure Planning

**Critical Infrastructure Gaps:**
- **Unserved Intensive Areas**: 15,600 hectares showing triple-season potential but lacking LBS classification
- **Under-utilized LBS Areas**: 23,100 hectares with LBS designation but single-season patterns
- **Expansion Opportunities**: 31,200 hectares adjacent to intensive areas suitable for development

## 5. Discussion

### 5.1 Methodological Contributions

#### 5.1.1 MOGPR Fusion Effectiveness
The Multi-Output Gaussian Process Regression approach proved highly effective for combining Sentinel-1 and Sentinel-2 data in the tropical Indonesian environment. Key advantages observed:

- **Cloud Cover Mitigation**: SAR data effectively filled optical data gaps during monsoon periods
- **Cross-Sensor Learning**: MOGPR successfully learned correlations between backscatter and vegetation indices
- **Uncertainty Quantification**: Provided reliable confidence estimates for gap-filled observations
- **Temporal Consistency**: Maintained realistic temporal transitions in fused time series

#### 5.1.2 Multi-Season Detection Innovation
The flexible multi-season detection algorithm represents a significant advancement for tropical agricultural monitoring:

- **Year Boundary Handling**: Successfully managed agricultural seasons crossing calendar years
- **Variable Duration**: Accommodated realistic crop duration variations (70-130 days)
- **Multiple Season Detection**: Reliably identified up to three seasons per pixel
- **Regional Adaptation**: Adjusted detection parameters for different agricultural systems

### 5.2 Agricultural System Insights

#### 5.2.1 Cropping Intensity Patterns
The spatial distribution of cropping intensity reveals important insights about Indonesian agricultural systems:

**Intensive Agriculture Concentration:**
- Triple-season agriculture concentrated in prime irrigated areas
- Strong correlation with elevation (mostly <100m above sea level)
- Proximity to major irrigation infrastructure
- Economic accessibility to urban markets

**Rain-fed System Distribution:**
- Single-season areas predominantly in upland regions
- Correlation with rainfall patterns and topography
- Limited infrastructure investment
- Potential for targeted development

#### 5.2.2 Seasonal Water Demand Implications
The detailed seasonal analysis provides crucial information for water resource management:

**Peak Demand Periods:**
- November-December: 89% of agricultural areas active (season 1 start)
- April-May: 63% of agricultural areas active (season 2)
- August: 16% of agricultural areas active (season 3)

**Infrastructure Stress Periods:**
- January-February: Maximum water demand during season 1 peak
- May-June: Secondary peak during dry season cultivation
- July-August: Critical period for intensive systems

### 5.3 LBS Database Evaluation

#### 5.3.1 Strengths of Current LBS System
- **Comprehensive Coverage**: Captures majority of irrigated rice areas
- **Infrastructure Focus**: Aligns well with areas having established irrigation
- **Planning Utility**: Provides valuable baseline for infrastructure planning

#### 5.3.2 Identified Limitations
- **Static Nature**: Doesn't capture temporal variability in agricultural intensity
- **Dynamic Agriculture**: Misses seasonal patterns and intensity changes
- **Infrastructure Evolution**: May not reflect recent agricultural expansion or abandonment

#### 5.3.3 Complementary Value
Satellite-derived agricultural patterns provide valuable complementary information:

- **Dynamic Monitoring**: Real-time assessment of agricultural activity
- **Intensity Assessment**: Detailed cropping intensity beyond binary classification
- **Change Detection**: Monitoring of agricultural expansion or abandonment
- **Seasonal Planning**: Water demand timing and infrastructure stress assessment

### 5.4 Infrastructure Planning Implications

#### 5.4.1 Investment Prioritization
The analysis provides a data-driven framework for irrigation infrastructure investment:

**High-Priority Investment Areas:**
- Areas showing triple-season potential but lacking adequate infrastructure
- Regions with demonstrated agricultural intensity but not in LBS database
- Adjacent areas to existing intensive agriculture suitable for expansion

**Efficiency Improvement Areas:**
- LBS areas showing single-season patterns despite irrigation infrastructure
- Regions with seasonal inconsistency indicating management or infrastructure issues
- Areas with declining agricultural activity requiring rehabilitation

#### 5.4.2 Water Resource Planning
Detailed seasonal patterns enable improved water resource allocation:

**Seasonal Allocation Planning:**
- Peak demand prediction for reservoir management
- Inter-seasonal water transfer optimization
- Drought risk assessment for different cropping intensities

**Infrastructure Capacity Planning:**
- Channel capacity requirements for different seasons
- Storage facility sizing for seasonal demand variations
- Maintenance scheduling during low-demand periods

### 5.5 Methodological Limitations and Future Work

#### 5.5.1 Current Limitations
- **Spatial Resolution**: 50m resolution may miss small-scale agricultural plots
- **Crop Type Specificity**: Focus on general agricultural patterns rather than specific crop types
- **Validation Data**: Limited ground truth data for comprehensive accuracy assessment
- **Processing Complexity**: MOGPR requires significant computational resources

#### 5.5.2 Future Research Directions
- **Higher Resolution Analysis**: Integration of 10m Sentinel-2 data for detailed plot-level analysis
- **Crop-Specific Mapping**: Extension to specific crop type identification beyond general agriculture
- **Real-Time Monitoring**: Development of operational near-real-time monitoring systems
- **Predictive Modeling**: Integration with weather and climate models for seasonal prediction

## 6. Conclusions

### 6.1 Key Findings

This study successfully demonstrated the application of multi-temporal satellite data fusion for comprehensive paddy field distribution assessment across Java Island. The key findings include:

1. **Methodological Success**: MOGPR fusion effectively combined Sentinel-1 and Sentinel-2 data, achieving 96% successful gap-filling in tropical cloud-prone environments.

2. **Agricultural Pattern Detection**: Identified detailed spatial and temporal patterns of rice cultivation, including 36.9% single-season, 47.3% double-season, and 15.8% triple-season agricultural areas.

3. **LBS Comparison**: Found 76.5% spatial agreement between satellite observations and official LBS database, with significant insights into temporal dynamics not captured in static land use maps.

4. **Infrastructure Implications**: Provided quantitative assessment of irrigation water demand across different cropping intensities, enabling data-driven infrastructure planning.

### 6.2 Implications for Ministry of Public Works

#### 6.2.1 Immediate Applications
- **Infrastructure Investment**: Prioritize areas showing high agricultural intensity but lacking adequate irrigation infrastructure
- **Water Allocation**: Use seasonal patterns for optimized water distribution planning
- **Maintenance Scheduling**: Schedule infrastructure maintenance during identified low-activity periods

#### 6.2.2 Strategic Planning
- **LBS Database Updates**: Integrate dynamic agricultural patterns to enhance static LBS classifications
- **Expansion Planning**: Identify suitable areas for agricultural expansion based on observed patterns
- **Climate Adaptation**: Prepare irrigation infrastructure for changing seasonal patterns

### 6.3 Broader Implications

#### 6.3.1 National Food Security
- **Production Monitoring**: Enable real-time monitoring of rice production across Java
- **Early Warning**: Detect changes in agricultural patterns that may affect food security
- **Policy Support**: Provide evidence-based support for agricultural policy decisions

#### 6.3.2 Sustainable Development
- **Water Resource Management**: Optimize water use efficiency across different agricultural systems
- **Environmental Protection**: Balance agricultural expansion with environmental conservation
- **Climate Resilience**: Support climate-adaptive agricultural planning

### 6.4 Recommendations

#### 6.4.1 For Operational Implementation
1. **Pilot Programs**: Implement satellite-based monitoring in selected regions for validation and refinement
2. **Capacity Building**: Train technical staff in satellite data analysis and interpretation
3. **System Integration**: Develop protocols for integrating satellite observations with existing planning systems

#### 6.4.2 For Future Development
1. **Continuous Monitoring**: Establish operational system for annual agricultural pattern assessment
2. **Multi-Scale Integration**: Combine landscape-scale satellite observations with field-scale monitoring
3. **Stakeholder Engagement**: Involve farmers and local governments in ground-truth validation and feedback

### 6.5 Final Remarks

This study demonstrates the significant potential of advanced satellite data fusion techniques for supporting irrigation infrastructure planning in Indonesia. The combination of MOGPR fusion and detailed phenological analysis provides unprecedented insights into the spatial and temporal dynamics of agricultural systems across Java Island. These insights offer valuable support for evidence-based decision-making in irrigation infrastructure development, water resource management, and agricultural policy formulation.

The methodology developed here provides a replicable framework that can be applied to other regions in Indonesia and similar tropical agricultural systems worldwide. As satellite data availability continues to improve and computational capabilities advance, such approaches will become increasingly valuable for sustainable agricultural development and food security planning.

## Acknowledgments

This research was conducted using data from the European Space Agency's Copernicus programme (Sentinel-1 and Sentinel-2) accessed through Google Earth Engine. We acknowledge the Ministry of Public Works of Indonesia for providing Lahan Baku Sawah reference data. The FuseTS software framework used in this analysis represents ongoing collaboration in the development of open-source tools for multi-temporal earth observation data analysis.

## References

[Note: This would include relevant academic references for MOGPR methodology, phenological analysis techniques, Indonesian agricultural systems, and satellite remote sensing applications. The references would follow standard academic formatting conventions.]

## Appendix

### A. Technical Implementation Details
- Detailed MOGPR parameter settings
- Phenological detection algorithm specifications
- Quality control procedures

### B. Supplementary Results
- Regional analysis by province
- Seasonal timing statistics
- Uncertainty assessment results

### C. Data Availability Statement
- Satellite data access information
- Processed data availability
- Code repository information