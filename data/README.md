# Data sources

## [37 Bus sytem](powerworld/glover37)

Design project 2: System planning for generation retirement

Glover, J. Duncan, Sarma, Mulukutla S. and Overbye, Thomas J. (2012). Power system analysis and design : Si Edition / J. Duncan Glover, Mulukutla S. Sarma, Thomas J. Overbye. 5. ed., SI. Stamford, CT: Cengage Learning.


## [SE3 demand series](se3_2025.csv)

`se3_2025.csv` contains the hourly 2025 demand series for the Swedish
SE3 bidding area used solely as the temporal scaling signal in the paper.

Source:
- Provider: [ENTSO-E transparency platform](https://transparency.entsoe.eu/)
- Dataset/product: Total load - Day-ahead/Actual (TR 6.1.B)
- Units: MW
- Timezone: UTC

Processing:
- No short-circuit results were used to select the temporal interval.
- The seven-day interval is selected programmatically as the continuous
  168-hour window with maximum Pmax-Pmin.