"""Transformer to parse and augment US zipcodes with info from zipcode database."""
from h2oaicore.transformer_utils import CustomTransformer
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning
import datatable as dt
import numpy as np

_global_modules_needed_by_name = ['pycodestyle==2.11.1', 'uszipcode==1.0.1']
from uszipcode import SearchEngine


class USZipcodeDBTransformer(CustomTransformer):
    _unsupervised = True

    _allow_transform_to_modify_output_feature_names = True

    _numeric_output = True

    @staticmethod
    def do_acceptance_test():
        return False

    @staticmethod
    def get_default_properties():
        return dict(col_type="categorical", min_cols=1, max_cols=1, relative_importance=1)

    @staticmethod
    def to_dict_values(data, name):
        result = dict()
        data = data[name]
        if data is None or len(data) == 0:
            return result
        for k in range(len(data)):
            key = data[k]['key']
            values = data[k]['values']
            names = [d['x'] for d in values]
            if len(data) > 1:
                keys = [name + '_' + key + '_' + str(y) for y in names]
            else:
                keys = [name + '_' + str(y) for y in names]
            vals = [d['y'] for d in values]
            result = {**result, **dict(zip(keys, vals))}
        return result

    @staticmethod
    def replaceBannedCharacters(str):
        return str.replace('<', ' less ').replace('[', '(').replace(']', ')')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transformer_name = self.__class__.__name__
        if self.transformer_name.endswith("Transformer"):
            self.transformer_name = self.transformer_name[:-len("Transformer")]

        self.search = SearchEngine(simple_zipcode=False)

    def get_zipcode_features(self, value):
        if value is None or not value:
            return self.get_zipcode_null_features()
        elif len(value) < 3:
            # raise ValueError("Value '{}' too short for zip code.".format(value))
            return self.get_zipcode_null_features()
        elif value[:5] in ['000', '0000', '00000', '   ', '    ', '     ']:
            return self.get_zipcode_null_features()
        else:
            lookup_value = value[:5]  # US zipcode only

        zip_data = self.search.by_zipcode(lookup_value)
        if (zip_data.zipcode_type == None):
            return self.get_zipcode_null_features()
            # raise ValueError("Value '{}' not a zipcode.".format(value))
        else:
            zip_dict = zip_data.to_dict()
            result = {  # 'zip_key': value,
                # 'zipcode_type': zip_dict['zipcode_type'],
                # 'major_city': zip_dict['major_city'],
                # 'post_office_city': zip_dict['post_office_city'],
                # 'common_city_list': zip_dict['common_city_list'][0],
                # 'county': zip_dict['county'],
                # 'state': zip_dict['state'],

                'lat': zip_dict['lat'],
                'lng': zip_dict['lng'],

                # 'timezone': zip_dict['timezone'],

                'radius_in_miles': zip_dict['radius_in_miles'],

                # 'area_code_list': ['469', '972'],

                'population': zip_dict['population'],
                'population_density': zip_dict['population_density'],
                'land_area_in_sqmi': zip_dict['land_area_in_sqmi'],
                'water_area_in_sqmi': zip_dict['water_area_in_sqmi'],
                'housing_units': zip_dict['housing_units'],
                'occupied_housing_units': zip_dict['occupied_housing_units'],
                'median_home_value': zip_dict['median_home_value'],
                'median_household_income': zip_dict['median_household_income'],
                'bounds_west': zip_dict['bounds_west'],
                'bounds_east': zip_dict['bounds_east'],
                'bounds_north': zip_dict['bounds_north'],
                'bounds_south': zip_dict['bounds_south'],

                # 'zipcode': zip_dict['zipcode']
            }

            return {**result,
                    **self.to_dict_values(zip_dict, 'population_by_year'),
                    **self.to_dict_values(zip_dict, 'population_by_age'),
                    **self.to_dict_values(zip_dict, 'population_by_gender'),
                    **self.to_dict_values(zip_dict, 'population_by_race'),
                    **self.to_dict_values(zip_dict, 'head_of_household_by_age'),
                    **self.to_dict_values(zip_dict, 'families_vs_singles'),
                    **self.to_dict_values(zip_dict, 'households_with_kids'),
                    **self.to_dict_values(zip_dict, 'children_by_age'),
                    **self.to_dict_values(zip_dict, 'housing_type'),
                    **self.to_dict_values(zip_dict, 'year_housing_was_built'),
                    **self.to_dict_values(zip_dict, 'housing_occupancy'),
                    **self.to_dict_values(zip_dict, 'vancancy_reason'),
                    **self.to_dict_values(zip_dict, 'owner_occupied_home_values'),
                    **self.to_dict_values(zip_dict, 'rental_properties_by_number_of_rooms'),
                    **self.to_dict_values(zip_dict, 'monthly_rent_including_utilities_studio_apt'),
                    **self.to_dict_values(zip_dict, 'monthly_rent_including_utilities_1_b'),
                    **self.to_dict_values(zip_dict, 'monthly_rent_including_utilities_2_b'),
                    **self.to_dict_values(zip_dict, 'monthly_rent_including_utilities_3plus_b'),
                    **self.to_dict_values(zip_dict, 'employment_status'),
                    **self.to_dict_values(zip_dict, 'average_household_income_over_time'),
                    **self.to_dict_values(zip_dict, 'household_income'),
                    **self.to_dict_values(zip_dict, 'annual_individual_earnings'),
                    **self.to_dict_values(zip_dict,
                                          'sources_of_household_income____percent_of_households_receiving_income'),
                    **self.to_dict_values(zip_dict,
                                          'sources_of_household_income____average_income_per_household_by_income_source'),
                    **self.to_dict_values(zip_dict,
                                          'household_investment_income____percent_of_households_receiving_investment_income'),
                    **self.to_dict_values(zip_dict,
                                          'household_investment_income____average_income_per_household_by_income_source'),
                    **self.to_dict_values(zip_dict,
                                          'household_retirement_income____percent_of_households_receiving_retirement_incom'),
                    **self.to_dict_values(zip_dict,
                                          'household_retirement_income____average_income_per_household_by_income_source'),
                    **self.to_dict_values(zip_dict, 'source_of_earnings'),
                    **self.to_dict_values(zip_dict, 'means_of_transportation_to_work_for_workers_16_and_over'),
                    **self.to_dict_values(zip_dict, 'travel_time_to_work_in_minutes'),
                    **self.to_dict_values(zip_dict, 'educational_attainment_for_population_25_and_over'),
                    **self.to_dict_values(zip_dict, 'school_enrollment_age_3_to_17')
                    }

    def get_zipcode_null_features(self):
        null_dict = self.get_zipcode_features('79936')
        for key, value in null_dict.items():
            null_dict[key] = None
        return null_dict

    def get_zipcode_null_result(self, X, original_zip_column_name):
        X[:, 'zip_key'] = '79936'
        zip_list = ['79936']
        zip_features = [self.get_zipcode_null_features() for x in zip_list]
        X_g = dt.Frame({"zip_key": zip_list})
        X_g.cbind(dt.Frame(zip_features))
        X_g.key = 'zip_key'
        X_result = X[:, :, dt.join(X_g)]
        self._output_feature_names = ["{}:{}.{}".format(self.transformer_name,
                                                        original_zip_column_name, self.replaceBannedCharacters(f)) for f
                                      in list(X_result[:, 1:].names)]
        self._feature_desc = ["Property '{}' of zipcode column ['{}'] from US zipcode database (recipe '{}')".format(
            f, original_zip_column_name, self.transformer_name) for f in list(X_result[:, 1:].names)]
        return X_result[:, 1:]

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        logger = None
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(experiment_id=self.context.experiment_id,
                                            tmp_dir=self.context.tmp_dir,
                                            experiment_tmp_dir=self.context.experiment_tmp_dir)

        X = dt.Frame(X)
        original_zip_column_name = X.names[0]
        X = X[:, dt.str64(dt.f[0])]
        X.names = ['zip_key']
        try:
            zip_list = dt.unique(X[~dt.isna(dt.f.zip_key), 0]).to_list()[0] + ['79936']
            zip_features = [self.get_zipcode_features(x) for x in zip_list]
            X_g = dt.Frame({"zip_key": zip_list})
            X_g.cbind(dt.Frame(zip_features))
            X_g.key = 'zip_key'
            X_result = X[:, :, dt.join(X_g)]
            self._output_feature_names = ["{}:{}.{}".format(self.transformer_name,
                                                            original_zip_column_name, self.replaceBannedCharacters(f))
                                          for f in list(X_result[:, 1:].names)]
            self._feature_desc = [
                "Property '{}' of zipcode column ['{}'] from US zipcode database (recipe '{}')".format(
                    f, original_zip_column_name, self.transformer_name) for f in list(X_result[:, 1:].names)]
            return X_result[:, 1:]
        except ValueError as ve:
            loggerinfo(logger, "Column '{}' is not a zipcode: {}".format(original_zip_column_name, str(ve)))
            return self.get_zipcode_null_result(X, original_zip_column_name)
        except TypeError as te:
            loggerwarning(logger, "Column '{}' triggered TypeError: {}".format(original_zip_column_name, str(te)))
            raise te
