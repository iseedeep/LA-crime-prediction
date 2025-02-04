CRIME_CATEGORY_MAPPING = {
    # Violent Crimes
    'CRIMINAL HOMICIDE': 'VIOLENT_CRIME',
    'MANSLAUGHTER, NEGLIGENT': 'VIOLENT_CRIME',
    'RAPE, FORCIBLE': 'VIOLENT_CRIME',
    'RAPE, ATTEMPTED': 'VIOLENT_CRIME',
    'ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT': 'VIOLENT_CRIME',
    'ASSAULT WITH DEADLY WEAPON ON POLICE OFFICER': 'VIOLENT_CRIME',
    'INTIMATE PARTNER - AGGRAVATED ASSAULT': 'VIOLENT_CRIME',
    'KIDNAPPING': 'VIOLENT_CRIME',
    'ROBBERY': 'VIOLENT_CRIME',
    'ATTEMPTED ROBBERY': 'VIOLENT_CRIME',
    
    # Property Crimes
    'BURGLARY': 'PROPERTY_CRIME',
    'BURGLARY FROM VEHICLE': 'PROPERTY_CRIME',
    'VEHICLE - STOLEN': 'PROPERTY_CRIME',
    'THEFT-GRAND ($950.01 & OVER)EXCPT,GUNS,FOWL,LIVESTK,PROD': 'PROPERTY_CRIME',
    'THEFT PLAIN - PETTY ($950 & UNDER)': 'PROPERTY_CRIME',
    'THEFT FROM MOTOR VEHICLE - GRAND ($950.01 AND OVER)': 'PROPERTY_CRIME',
    'THEFT FROM MOTOR VEHICLE - PETTY ($950 & UNDER)': 'PROPERTY_CRIME',
    'BURGLARY, ATTEMPTED': 'PROPERTY_CRIME',
    'VANDALISM - FELONY ($400 & OVER, ALL CHURCH VANDALISMS)': 'PROPERTY_CRIME',
    'VANDALISM - MISDEAMEANOR ($399 OR UNDER)': 'PROPERTY_CRIME',
    
    # Sexual Crimes
    'BATTERY WITH SEXUAL CONTACT': 'SEXUAL_CRIME',
    'ORAL COPULATION': 'SEXUAL_CRIME',
    'SEXUAL PENETRATION W/FOREIGN OBJECT': 'SEXUAL_CRIME',
    'LEWD/LASCIVIOUS ACTS WITH CHILD': 'SEXUAL_CRIME',
    'CHILD PORNOGRAPHY': 'SEXUAL_CRIME',
    
    # Domestic/Family Violence
    'INTIMATE PARTNER - SIMPLE ASSAULT': 'DOMESTIC_VIOLENCE',
    'VIOLATION OF RESTRAINING ORDER': 'DOMESTIC_VIOLENCE',
    'VIOLATION OF COURT ORDER': 'DOMESTIC_VIOLENCE',
    'VIOLATION OF TEMPORARY RESTRAINING ORDER': 'DOMESTIC_VIOLENCE',
    
    # Weapons/Firearms
    'WEAPONS POSSESSION/BOMBING': 'WEAPONS_OFFENSE',
    'DISCHARGE FIREARMS/SHOTS FIRED': 'WEAPONS_OFFENSE',
    'SHOTS FIRED AT INHABITED DWELLING': 'WEAPONS_OFFENSE',
    'BRANDISH WEAPON': 'WEAPONS_OFFENSE',
    
    # Fraud/Financial
    'CREDIT CARDS, FRAUD USE ($950.01 & OVER)': 'FRAUD',
    'THEFT OF IDENTITY': 'FRAUD',
    'DOCUMENT FORGERY / STOLEN FELONY': 'FRAUD',
    'COUNTERFEIT': 'FRAUD',
    'EMBEZZLEMENT, GRAND THEFT ($950.01 & OVER)': 'FRAUD',
    
    # Public Order
    'DISTURBING THE PEACE': 'PUBLIC_ORDER',
    'TRESPASSING': 'PUBLIC_ORDER',
    'DRINKING IN PUBLIC': 'PUBLIC_ORDER',
    'ILLEGAL DUMPING': 'PUBLIC_ORDER',
    
    # Default category for all other crimes
    'DEFAULT': 'OTHER'
}