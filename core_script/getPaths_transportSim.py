from abc import ABC, abstractmethod
import pandas as pd

from core_script.TransportSim_utils import *

warnings.filterwarnings('ignore')


# define a decorator helper method to time function runtime
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print("It takes {:.3f} seconds to run the function '{}()'!".format((te - ts), method.__name__))
            # print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed


# base abstract class defining methods/functions to be implemented in its subclasses
class TravelSim(ABC):
    @abstractmethod
    def prepare_network(self):
        pass

    @property
    @abstractmethod
    def dict_settings(self):
        pass

    @dict_settings.setter
    @abstractmethod
    def dict_settings(self, dict_settings):
        pass

    @abstractmethod
    def run_all(self):
        pass

    # this method will be directly inherited by the subclasses
    @timeit
    def prepare_trips(self, option=None):
        if option is None:
            option = self.options[0]
        os.chdir(self.trip_lib)  # change to trip input output directory
        print('** load trip data & prepare sample **')
        df_points = pd.DataFrame()
        # Section 2: lOAD SAMPLE DATA; could have multiple csv sample files.
        if self.dict_settings['one_by_one'] is False:
            if self.trips_prepared:
                for samp_file in glob.glob(os.path.join('samples_out', '*.csv')):
                    dfi = pd.read_csv(samp_file)
                    dfi['sample'] = os.path.basename(samp_file).split('.csv')[0]
                    df_points = df_points.append(dfi, ignore_index=True)
            else:
                for samp_file in glob.glob(os.path.join('samples_in', '*.csv')):
                    df_points = df_points.append(samp_pre_process(samp_file, self.dict_settings, option=option),
                                                 ignore_index=True)
        else:
            if self.trips_prepared:
                for samp_file in glob.glob(os.path.join('one_by_one_out', '*.csv')):
                    dfi = pd.read_csv(samp_file)
                    dfi['sample'] = os.path.basename(samp_file).split('.csv')[0]
                    df_points = df_points.append(dfi, ignore_index=True)
            else:
                for samp_file in glob.glob(os.path.join('one_by_one_in', '*.csv')):
                    df_points = df_points.append(samp_pre_process(samp_file, self.dict_settings, option=option),
                                                 ignore_index=True)
        self.df_points = df_points
        self.trips_prepared = True  # means found samples_out file with nearest node found!
        return df_points


# write SidewalkSim subclass for sidewalk queries
class BikewaySim(TravelSim):
    def __init__(self, proj_lib, network_lib, trip_lib, network_prepared=False, trips_prepared=False):
        print('Initialize BikewaySim object to process sidewalk graph and pre-process trip queries!')
        self.proj_lib = proj_lib
        self.graph_lib = network_lib
        self.trip_lib = trip_lib
        self.options = ['bike']

        self.network_prepared = network_prepared
        self.dict_settings = None
        self.trips_prepared = trips_prepared
        self.df_points = None

    @timeit
    def prepare_network(self, grid_size=10000.0):
        if self.network_prepared:
            print('** Load BikewaySim network computed before **')
            df_links = gpd.read(os.path.join(os.environ['PROJ_LIB'], 'trips_bws',
                                             'data_node_link', 'links.shp'))
        else:
            print('** Initialize BikewaySim network **')
            # change to graph directory
            dir_path = os.path.join(self.proj_lib, 'build_graph')
            os.chdir(dir_path)
            df_links = initialize_bikewaysim_links()
            # save file to .gdb for reuse
            df_links.to_file(os.path.join(os.environ['PROJ_LIB'], 'trips_bws',
                                          'data_node_link', 'links.shp'))
            # prepare graph
            dict_walk = {'DG': build_bike_network(df_links), 'links': df_links}
        return df_links, dict_walk

    @property
    def dict_settings(self):
        # print('get TravelSim dict_settings!')
        return self._dict_settings

    @dict_settings.setter
    def dict_settings(self, dict_settings):
        print('Set/Update BikewaySim dict_settings!')
        self._dict_settings = dict_settings

    @timeit
    def run_all(self):
        print('**for each trip, find k-shortest paths**')
        os.chdir(self.trip_lib)
        allRun(self.df_points, self.options, self.dict_settings)


# write SidewalkSim subclass for sidewalk queries
class SidewalkSim(TravelSim):
    def __init__(self, proj_lib, data_lib, trip_lib, network_prepared=False, trips_prepared=False):
        print('Initialize SidewalkSim object to process sidewalk graph and pre-process trip queries!')
        self.proj_lib = proj_lib
        self.graph_lib = data_lib
        self.trip_lib = trip_lib
        self.options = ['sidewalk']

        self.network_prepared = network_prepared
        self.dict_settings = None
        self.trips_prepared = trips_prepared
        self.df_points = None

    @timeit
    def prepare_network(self, grid_size=10000.0):
        if self.network_prepared:
            print('** Load SidewalkSim network computed before **')
            df_links = gpd.read_file(os.path.join(os.environ['PROJ_LIB'], 'trips_bws',
                                                  'data_node_link', 'SWS_links.shp'))
        else:
            print('** Initialize sidewalk network **')
            # change to graph directory
            dir_path = os.path.join(self.proj_lib, 'build_graph')
            os.chdir(dir_path)
            df_links = initialize_sws_links()
            # save file to .gdb for reuse
            df_links.to_file(os.path.join(os.environ['PROJ_LIB'], 'trips_sws',
                                          'data_node_link', 'SWS_links.shp'))
            # prepare graph
            dict_walk = {'DG': build_walk_network(df_links), 'links': df_links}
            self.network_prepared = True
        return df_links, dict_walk

    @property
    def dict_settings(self):
        # print('get TravelSim dict_settings!')
        return self._dict_settings

    @dict_settings.setter
    def dict_settings(self, dict_settings):
        print('Set/Update SidewalkSim dict_settings!')
        self._dict_settings = dict_settings

    def run_all(self):
        print('**for each trip, find k-shortest paths**')
        os.chdir(self.trip_lib)
        allRun(self.df_points, self.options, self.dict_settings)


def main_SidewalkSim(sws):
    df_links, dict_walk = sws.prepare_network()
    # strategy: 1. given origin time find earliest arrival
    #           2. given expected arrival time find latest departure time
    dict_settings = {'walk_speed': 2.0,  # people's walking speed 2 mph
                     'grid_size': 25000.0,
                     # for searching nearby links by grouping links to grids with width 25000 ft.
                     # for better efficiency in searching
                     'ntp_dist_thresh': 5280,
                     # node to point (maximum distance access to network from origin/destination);
                     'network': {'sidewalk': dict_walk},  # dump in networks and modes
                     # strategy determines network link's direction.
                     # Strategy 1: Find earliest arrival given query time as departure time
                     # Strategy 2: Find latest departure time given query time as arrival time
                     'strategy': {'sidewalk': 1},  # 1. find earliest arrival 2. find latest departure
                     'query_time': [8],  # departure time or arrival time of a trip, depends on the strategy

                     'walk_thresh': {'sidewalk': 1},  # walking threshold is 1 mile (IGNORED by the Sidewalk option)
                     'num_options': {'sidewalk': 1},  # if set to 2, return 2 shortest paths
                     'plot_all': True,  # if True, plot results and save plots for all routes found
                     'one_by_one': False  # set time and strategy one by one
                     }
    # load dict_settings to the sws object
    sws.dict_settings = dict_settings
    __ = sws.prepare_trips(option='sidewalk')
    sws.run_all()


def main_BikewaySim(bws):
    df_links, dict_bike = bws.prepare_network()

    # strategy: 1. given origin time find earliest arrival
    #           2. given expected arrival time find latest departure time
    dict_settings = {'walk_speed': 2.0,  # people's walking speed 2 mph
                     'grid_size': 25000.0,
                     # for searching nearby links by grouping links to grids with width 25000 ft.
                     # for better efficiency in searching
                     'ntp_dist_thresh': 5280.0,
                     # node to point (maximum distance access to network from origin/destination);
                     # a (walking) distance threshold
                     'network': {'bike': dict_bike},  # dump in networks and modes
                     # strategy determines network link's direction.
                     # Strategy 1: Find earliest arrival given query time as departure time
                     # Strategy 2: Find latest departure time given query time as arrival time
                     'strategy': {'bike': 1},  # 1. find earliest arrival 2. find latest departure
                     'query_time': [8],  # departure time or arrival time of a trip, depends on the strategy

                     'walk_thresh': {'bike': 1},  # walking threshold is 1 mile
                     'num_options': {'bike': 1},  # if set to 2, return 2-shortest paths
                     'plot_all': True,  # if True, plot results and save plots for all routes found
                     'one_by_one': False  # set time and strategy one by one
                     }
    # load dict_settings to the sws object
    bws.dict_settings = dict_settings
    __ = bws.prepare_trips(option='bike')
    bws.run_all()


if __name__ == '__main__':
    # I prefer to use Jupyter notebook running it in several blocks,
    # instead of running the file at once.
    # The notebook can be set to auto-load this file/module every time
    # you make changes in this module and saved it.
    # But one can also just run the file directly through Terminal/Command Lines.

    # need to set this environmental path for network data and query data at separate locations
    # set path variable for SidewalkSim
    os.environ['PROJ_LIB'] = '/Users/diyi93/Desktop/gra/TransportSim'

    # network shapefile data path directory
    os.environ['sws_NETWORK'] = '/Users/diyi93/Desktop/gra/TransportSim/sidewalk_raw_files'
    os.environ['bws_NETWORK'] = '/Users/diyi93/Desktop/gra/TransportSim/ABM2020 203K'

    # trip queries directory (store trip inputs and trip outputs and network results)
    os.environ['sws_TRIPS'] = '/Users/diyi93/Desktop/gra/TransportSim/trips_sws'
    os.environ['bws_TRIPS'] = '/Users/diyi93/Desktop/gra/TransportSim/trips_bws'

    # create SidewalkSim object for running 'sidewalk' option
    sws = SidewalkSim(os.environ['PROJ_LIB'], os.environ['sws_NETWORK'], os.environ['sws_TRIPS'])
    # create BikewaySim object for running 'bikewaysim' option
    bws = BikewaySim(os.environ['PROJ_LIB'], os.environ['bws_NETWORK'], os.environ['bws_TRIPS'])

    main_BikewaySim(bws)


