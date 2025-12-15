WITH census_data AS (
    SELECT
        LEFT(a.CENSUS_BLOCK_GROUP, 11) as census_tract,
        SUM(a."B15002e1") AS total_population_25plus,
        SUM(a."B15002e15") AS male_bachelors_degree,
        SUM(a."B15002e32") AS female_bachelors_degree,
        SUM(c."B03002e1") AS total_population,
        SUM(c."B03002e3") AS non_hispanic_white_population,
        CASE WHEN SUM(c."B03002e1") > 0
             THEN (SUM(c."B03002e3") / SUM(c."B03002e1")) * 100
             ELSE NULL END AS pct_white,
        AVG(d."B20002e1") AS median_earnings_total,
        AVG(d."B20002e2") AS median_earnings_male,
        AVG(d."B20002e3") AS median_earnings_female
    FROM US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET.PUBLIC."2019_CBG_B15" AS a
    LEFT JOIN US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET.PUBLIC."2019_CBG_B03" AS c
        ON a.CENSUS_BLOCK_GROUP = c.CENSUS_BLOCK_GROUP
    LEFT JOIN US_OPEN_CENSUS_DATA__NEIGHBORHOOD_INSIGHTS__FREE_DATASET.PUBLIC."2019_CBG_B20" AS d
        ON a.CENSUS_BLOCK_GROUP = d.CENSUS_BLOCK_GROUP
    WHERE LEFT(a.CENSUS_BLOCK_GROUP, 2) IN ('37', '51', '34', '36', '45', '39', '24')
    GROUP BY LEFT(a.CENSUS_BLOCK_GROUP, 11)
),
school_metrics AS (
    SELECT
        nb.ID as neighborhood_id,
        nb.STATE as state_code,
        AVG(s.STUDENT_TEACHER_RATIO) as nbhd_avg_student_teacher_ratio,
        AVG(s.ENROLLMENT) as nbhd_avg_school_size,
        COUNT(DISTINCT CASE WHEN s.SCHOOL_RATING = 'Above Average' THEN s.ID END) as nbhd_above_avg_schools_cnt,
        COUNT(DISTINCT CASE WHEN s.ELEMENTARY_SCHOOL_IND = 'Yes' AND s.STATUS = 'Open' THEN s.ID END) as nbhd_elementary_schools_cnt,
        COUNT(DISTINCT CASE WHEN s.MIDDLE_SCHOOL_IND = 'Yes' AND s.STATUS = 'Open' THEN s.ID END) as nbhd_middle_schools_cnt,
        COUNT(DISTINCT CASE WHEN s.HIGH_SCHOOL_IND = 'Yes' AND s.STATUS = 'Open' THEN s.ID END) as nbhd_high_schools_cnt,
        AVG(CASE WHEN s.ENROLLMENT > 0 THEN (s.ENROLL_FREE_OR_REDUCED_LUNCH / s.ENROLLMENT) * 100 END) as nbhd_avg_pct_free_reduced_lunch,
        COUNT(DISTINCT CASE WHEN s.AP_IND = 'Yes' THEN s.ID END) as nbhd_ap_schools_cnt,
        COUNT(DISTINCT CASE WHEN s.GIFTED_AND_TALENTED_PROG_IND = 'Yes' THEN s.ID END) as nbhd_gifted_prog_schools_cnt,
        AVG(s.PREVIOUS_YEAR_GRADE_12_TO_4_YEAR_COLLEGE_PCT) as nbhd_avg_college_going_rate
    FROM
        ATTOM_HOUSE_IQ_SHARE.DELIVERY.NEIGHBORHOODS_LEV_2 nb
    LEFT JOIN
        ATTOM_HOUSE_IQ_SHARE.DELIVERY.SCHOOLS s
        ON ST_DISTANCE(
            TO_GEOGRAPHY(nb.GEOMETRY),
            TO_GEOGRAPHY(ST_POINT(s.LONGITUDE, s.LATITUDE))
        ) <= 8047  -- 5 miles in meters
        AND s.STATE_CD = nb.STATE  -- Now both are 2-letter codes
        AND s.STATUS = 'Open'
    WHERE nb.STATE IN ('NC', 'VA', 'NJ', 'NY', 'SC', 'OH', 'MD')  -- Changed to uppercase codes
    GROUP BY nb.ID, nb.STATE
)
SELECT
    p.PROPERTYID as property_id,
    p.CURRENTSALESPRICE as sale_price,
    p.CURRENTSALERECORDINGDATE as sale_date,
    p.YEARBUILT as year_built,
    p.EFFECTIVEYEARBUILT as effective_year_built,
    p.SITUSLATITUDE as latitude,
    p.SITUSLONGITUDE as longitude,
    p.SITUSSTATE as state,
    p.SITUSCITY as city,
    p.SITUSZIP5 as zip,
    p.SITUSCENSUSTRACT as census_tract,
    p.SUMLIVINGAREASQFT as living_sqft,
    p.LOTSIZESQFT as lot_sqft,
    p.BEDROOMS as bedrooms,
    p.BATHFULL as full_baths,
    p.BATHSPARTIALNBR as half_baths,
    p.GARAGEPARKINGNBR as garage_spaces,
    p.FIREPLACECODE as fireplace_code,

    -- Census Demographics (Education)
    c.total_population_25plus,
    c.male_bachelors_degree,
    c.female_bachelors_degree,
    c.total_population,
    c.non_hispanic_white_population,
    c.pct_white,

    -- Census Demographics (Income)
    c.median_earnings_total,
    c.median_earnings_male,
    c.median_earnings_female,

    -- Election Data
    v.county_name,
    v.county_fips,
    v.votes_gop,
    v.votes_dem,
    v.total_votes,
    v.per_gop,
    v.per_dem,
    v.per_point_diff,
    v.dem_margin,
    v.rep_margin,

    -- Neighborhood Info - Core Demographics
    n.GEO_NM as neighborhood_name,
    n.POPULATION as nbhd_population,
    n.POPULATION_DENSITY_SQ_MI as nbhd_pop_density,
    n.MEDIAN_AGE as nbhd_median_age,
    n.MEDIAN_HOUSEHOLD_SIZE as nbhd_household_size,
    n.POPULATION_AGED_0_5_PCT as nbhd_pct_age_0_5,
    n.POPULATION_AGED_6_11_PCT as nbhd_pct_age_6_11,
    n.POPULATION_AGED_12_17_PCT as nbhd_pct_age_12_17,
    (COALESCE(n.POPULATION_AGED_0_5_PCT, 0) +
     COALESCE(n.POPULATION_AGED_6_11_PCT, 0) +
     COALESCE(n.POPULATION_AGED_12_17_PCT, 0)) as nbhd_pct_children,

    -- Neighborhood Info - Economic Profile
    n.MEDIAN_HOUSEHOLD_INCOME as nbhd_median_income,
    n.AVG_HOUSEHOLD_INCOME as nbhd_avg_income,
    n.HOUSEHOLD_INCOME_PER_CAPITA as nbhd_per_capita_income,
    n.POPULATION_IN_POVERTY_PCT as nbhd_poverty_rate,
    n.HOUSEHOLDS_INCOME_200000_AND_OVER_PCT as nbhd_pct_high_income,
    n.CPI as nbhd_cost_of_living_index,
    n.CPI_HOUSING as nbhd_housing_cost_index,

    -- Neighborhood Info - Housing Market
    n.HOUSING_OWNER_HOUSEHOLDS_MEDIAN_VALUE as nbhd_median_home_value,
    n.HOUSING_MEDIAN_RENT as nbhd_median_rent,
    n.HOUSING_UNITS_OWNER_OCCUPIED_PCT as nbhd_homeownership_rate,
    n.HOUSING_MEDIAN_BUILT_YR as nbhd_median_year_built,
    n.HOUSING_UNITS_VACANT_PCT as nbhd_vacancy_rate,

    -- Neighborhood Info - Education
    n.EDUCATION_BACH_DEGREE_PCT as nbhd_pct_bachelors,
    n.EDUCATION_GRAD_DEGREE_PCT as nbhd_pct_grad_degree,
    n.EDUCATION_HS_PCT as nbhd_pct_high_school,

    -- Neighborhood Info - Safety & Environment
    n.CRIME_INDEX as nbhd_crime_index,
    n.AIR_POLLUTION_INDEX as nbhd_air_quality_index,
    n.OZONE_INDEX as nbhd_ozone_index,

    -- Neighborhood Info - Commute & Lifestyle
    n.MEDIAN_TRAVEL_TIME_TO_WORK_MI as nbhd_median_commute_min,
    n.TRANSPORTATION_WORK_FROM_HOME_PCT as nbhd_pct_work_from_home,
    n.TRANSPORTATION_CAR_ALONE_PCT as nbhd_pct_drive_alone,
    n.TRANSPORTATION_PUBLIC_PCT as nbhd_pct_public_transit,

    -- Neighborhood Info - Employment
    n.OCCUPATION_WHITE_COLLAR_PCT as nbhd_pct_white_collar,
    n.EMPLOYEE_PROFESSIONAL_SCIENTIFIC_TECHNICAL_SVCS_NAICS_PCT as nbhd_pct_professional,
    n.EMPLOYEE_HEALTH_CARE_SOCIAL_ASSISTANCE_NAICS_PCT as nbhd_pct_healthcare,

    -- Housing Stock Quality Indicators
    n.HOUSING_BUILT_2010_OR_LATER_PCT as nbhd_pct_new_housing,
    n.HOUSING_OCCUPIED_STRUCTURE_1_UNIT_DETACHED_PCT as nbhd_pct_single_family,
    n.HOUSING_OCCUPIED_STRUCTURE_50_OR_MORE_UNITS_PCT as nbhd_pct_large_apartments,

    -- Family composition
    n.HOUSEHOLDS_FAMILY_MARRIED_PCT as nbhd_pct_married_families,
    n.HOUSEHOLDS_FAMILY_W_CHILDREN_PCT as nbhd_pct_families_with_children,

    -- School Quality Metrics (Aggregated)
    sm.nbhd_avg_student_teacher_ratio,
    sm.nbhd_avg_school_size,
    sm.nbhd_above_avg_schools_cnt,
    sm.nbhd_elementary_schools_cnt,
    sm.nbhd_middle_schools_cnt,
    sm.nbhd_high_schools_cnt,
    sm.nbhd_avg_pct_free_reduced_lunch,
    sm.nbhd_ap_schools_cnt,
    sm.nbhd_gifted_prog_schools_cnt,
    sm.nbhd_avg_college_going_rate,

    -- School District Financial Metrics
    d.TOTAL_PER_PUPIL_EXPENDITURE_AMT as district_per_pupil_spending,
    d.PER_PUPIL_EXP_INSTR_PCT as district_pct_spending_instruction,
    CASE WHEN d.ENROLLMENT > 0 THEN (d.TEACHER_CNT_FTE / d.ENROLLMENT) * 1000 END as district_teachers_per_1000_students,
    d.SCHOOL_DISTRICT_RATING as district_rating

FROM
    roc_public_record_data."DATATREE"."ASSESSOR" p

LEFT JOIN
    census_data c
    ON p.SITUSCENSUSTRACT = c.census_tract

LEFT JOIN
    "SCRATCH"."DATASCIENCE"."VOTING_PATTERNS_2020" v
    ON LEFT(p.FIPS, 5) = CAST(v.county_fips AS VARCHAR)

LEFT JOIN
    ATTOM_HOUSE_IQ_SHARE.DELIVERY.NEIGHBORHOODS_LEV_2 nb
    ON ST_CONTAINS(nb.GEOMETRY, TO_GEOMETRY(ST_POINT(p.SITUSLONGITUDE, p.SITUSLATITUDE)))
    AND nb.STATE IN ('NC', 'VA', 'NJ', 'NY', 'SC', 'OH', 'MD')  -- Changed to uppercase codes

LEFT JOIN
    ATTOM_HOUSE_IQ_SHARE.DELIVERY.COMMUNITY_INFO_NEIGHBORHOODS_LEV_2 n
    ON nb.ID = n.GEO_ID

LEFT JOIN
    school_metrics sm
    ON nb.ID = sm.neighborhood_id

LEFT JOIN
    ATTOM_HOUSE_IQ_SHARE.DELIVERY.SCHOOL_DISTRICT_BOUNDARIES sdb
    ON ST_CONTAINS(sdb.GEOMETRY, TO_GEOMETRY(ST_POINT(p.SITUSLONGITUDE, p.SITUSLATITUDE)))

LEFT JOIN
    ATTOM_HOUSE_IQ_SHARE.DELIVERY.SCHOOL_DISTRICTS d
    ON sdb.NCESDISTID = d.NCES_SCHOOL_DISTRICT_ID

WHERE
    p.SITUSSTATE IN ('nc', 'va', 'nj', 'ny', 'sc', 'oh', 'md')
    AND p.CURRENTSALESPRICE IS NOT NULL
    AND p.SUMLIVINGAREASQFT IS NOT NULL
    AND p.LOTSIZESQFT IS NOT NULL
    AND p.SITUSLATITUDE IS NOT NULL
    AND p.SITUSLONGITUDE IS NOT NULL
    AND n.CPI IS NOT NULL;