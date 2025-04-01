# Disease Parameters for SEIR models


disease_params = {'varicella': {'incubation_period': [15, 10, 21], # https://www.cdc.gov/pinkbook/hcp/table-of-contents/chapter-22-varicella.html
                                'infection_period': [5], # https://www.cdph.ca.gov/Programs/CID/DCDC/CDPH%20Document%20Library/Immunization/Varicella-Quicksheet.pdf
                                'vaccine_efficacy': [98, 82, 99], # https://www.cdc.gov/pinkbook/hcp/table-of-contents/chapter-22-varicella.html and 
                                'r0': [11, 10, 12], # https://transportgeography.org/contents/applications/transportation-pandemics/basic-reproduction-number-r0-of-major-infectious-diseases/
                                'fatality_rate': [.05/1000000]}, # https://pmc.ncbi.nlm.nih.gov/articles/PMC4514403/
                  'rubella': {'incubation_period': [14, 12, 23], # https://www.cdc.gov/pinkbook/hcp/table-of-contents/chapter-20-rubella.html
                                'infection_period': [], # https://www.researchgate.net/publication/351105137_THE_DYNAMICS_OF_RUBELLA_VIRUS_WITH_TWO-DOSE_VACCINATION_STRATEGY
                                'vaccine_efficacy': [95, 85], # https://www.cdc.gov/pinkbook/hcp/table-of-contents/chapter-20-rubella.html and https://www.researchgate.net/publication/351105137_THE_DYNAMICS_OF_RUBELLA_VIRUS_WITH_TWO-DOSE_VACCINATION_STRATEGY
                                'r0': [6, 5, 7], # https://transportgeography.org/contents/applications/transportation-pandemics/basic-reproduction-number-r0-of-major-infectious-diseases/
                                'fatality_rate': []},
                  'polio': {'incubation_period': [4.5, 3, 6], # https://www.cdc.gov/pinkbook/hcp/table-of-contents/chapter-18-poliomyelitis.html
                                'infection_period': [],
                                'vaccine_efficacy': [99, 90, 99.9], # https://www.cdc.gov/pinkbook/hcp/table-of-contents/chapter-18-poliomyelitis.html
                                'r0': [4], # https://academic.oup.com/jid/article/229/4/1097/7246204#446959537 ????
                                'fatality_rate': [3.5, 2, 5]}, # for children, higher for adolescents and adults https://www.cdc.gov/pinkbook/hcp/table-of-contents/chapter-18-poliomyelitis.html
                  'mumps': {'incubation_period': [17, 12, 25], # https://www.cdc.gov/pinkbook/hcp/table-of-contents/chapter-15-mumps.html
                                'infection_period': [],
                                'vaccine_efficacy': [88, 78], # https://www.cdc.gov/pinkbook/hcp/table-of-contents/chapter-15-mumps.html
                                'r0': [4.79, 3, 7], # https://pmc.ncbi.nlm.nih.gov/articles/PMC5899613/
                                'fatality_rate': []},
                  'measles': {'incubation_period': [11.5, 7, 21], # https://www.cdc.gov/pinkbook/hcp/table-of-contents/chapter-13-measles.html
                                'infection_period': [5],
                                'vaccine_efficacy': [99.95],
                                'r0': [15, 12, 18], # https://transportgeography.org/contents/applications/transportation-pandemics/basic-reproduction-number-r0-of-major-infectious-diseases/
                                'fatality_rate': [.0015]},
                  'hepatitis_b': {'incubation_period': [75, 60, 90], # https://www.cdc.gov/pinkbook/hcp/table-of-contents/chapter-10-hepatitis-b.html
                                'infection_period': [],
                                'vaccine_efficacy': [90, 80, 100], # https://www.cdc.gov/pinkbook/hcp/table-of-contents/chapter-10-hepatitis-b.html
                                'r0': [],
                                'fatality_rate': [.53/100000, .51/100000, .64/100000]}, # https://www.cdc.gov/hepatitis/statistics/2020surveillance/hepatitis-b/figure-2.8.htm
                  'diphtheria': {'incubation_period': [3.5, 1, 10], # https://www.cdc.gov/pinkbook/hcp/table-of-contents/chapter-7-diphtheria.html
                                'infection_period': [],
                                'vaccine_efficacy': [97], # https://www.cdc.gov/pinkbook/hcp/table-of-contents/chapter-7-diphtheria.html
                                'r0': [],
                                'fatality_rate': [7.5, 5, 10]}, # https://www.cdc.gov/pinkbook/hcp/table-of-contents/chapter-7-diphtheria.html
                  'pertussis': {'incubation_period': [8.5, 4, 21], #  https://www.cdc.gov/pinkbook/hcp/table-of-contents/chapter-16-pertussis.htm
                                'infection_period': [26], # https://pubmed.ncbi.nlm.nih.gov/21071671/ ?
                                'vaccine_efficacy': [82.5, 80, 85], #  https://www.cdc.gov/pinkbook/hcp/table-of-contents/chapter-16-pertussis.htm
                                'r0': [11, 9.9, 11.5], # https://pmc.ncbi.nlm.nih.gov/articles/PMC4408109/
                                'fatality_rate': [.000555, .0002822, .0013141]}} # https://www.cdc.gov/pertussis/php/surveillance/index.html