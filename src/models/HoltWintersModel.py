from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np

class HoltWintersModel:
    def __init__(self,
                    seasonality,
                    trend,
                    seasonal_period,
                    damped_trend=False,
                    initialization_method="estimated",
                    initial_level=None,
                    initial_trend=None,
                    initial_seasonal=None,
                    use_boxcox=False,
                    bounds=None,
                    dates=None,
                    freq=None,
                    missing='none'):
        """
        Initialize the Holt-Winters model with the given hyperparameters.

        Parameters:
        - seasonality (str): The type of seasonality. Valid values are 'add', 'mul', 'additive', 'multiplicative', or None.
        - trend (str): The type of trend. Valid values are 'add', 'mul', 'additive', 'multiplicative', or None.
        - seasonal_period (int): The number of time steps in a seasonal period.
        - damped_trend (bool): Whether to use a damped trend. Default is False.
        - initialization_method (str): The method to initialize the model. Valid values are 'estimated', 'heuristic', 'legacy-heuristic', or 'known'. Default is 'estimated'.
        - initial_level (float): The initial level value. Default is None.
        - initial_trend (float): The initial trend value. Default is None.
        - initial_seasonal (float): The initial seasonal value. Default is None.
        - use_boxcox (bool, float, str): Whether to use Box-Cox transformation. Valid values are True, False, a float value, or 'log'. Default is False.
        - bounds (None or tuple): The bounds for the parameters optimization. Default is None.
        - dates (None or array-like): The dates associated with the time series. Default is None.
        - freq (None or str): The frequency of the time series. Default is None.
        - missing (str): The method to handle missing values. Valid values are 'none', 'drop', or 'interpolate'. Default is 'none'.
        """
        # Check if seasonality is a string and validate its value
        if not isinstance(seasonality, str):
            raise TypeError("seasonality must be a string")
        if seasonality not in ['add', 'mul', 'additive', 'multiplicative', None]:
            raise ValueError("Invalid value for seasonality. Expected 'add', 'mul', 'additive', 'multiplicative', or None.")
        
        # Check if trend is a string and validate its value
        if not isinstance(trend, str):
            raise TypeError("trend must be a string")
        if trend not in ['add', 'mul', 'additive', 'multiplicative', None]:
            raise ValueError("Invalid value for trend. Expected 'add', 'mul', 'additive', 'multiplicative', or None.")
        
        # Check if seasonal_period is an integer and validate its value
        if not isinstance(seasonal_period, int):
            raise TypeError("seasonal_period must be an integer")
        
        # Check if damped_trend is a boolean
        if not isinstance(damped_trend, bool):
            raise TypeError("damped_trend must be a boolean")
        
        # Check if initialization_method is a string and validate its value
        if not isinstance(initialization_method, str):
            raise TypeError("initialization_method must be a string")
        if initialization_method not in ['estimated', 'heuristic', 'legacy-heuristic', 'known']:
            raise ValueError("Invalid value for initialization_method. Expected 'estimated', 'heuristic', 'legacy-heuristic', or 'known'")
        
        # Check if initial_level is a float
        if not isinstance(initial_level, float) and initial_level is not None:
            raise TypeError("initial_level must be a float")
        
        # Check if initial_trend is a float
        if not isinstance(initial_trend, float) and initial_trend is not None:
            raise TypeError("initial_trend must be a float")
        
        # Check if initial_seasonal is a float
        if not isinstance(initial_seasonal, float) and initial_seasonal is not None:
            raise TypeError("initial_seasonal must be a float")
        
        # Check if use_boxcox is either a boolean or float or has a value of 'log'
        if not isinstance(use_boxcox, (bool, float, str)):
            raise TypeError("use_boxcox must be a boolean, float or string")
        if isinstance(use_boxcox, str) and use_boxcox != 'log':
            raise ValueError("Invalid value for use_boxcox. Expected True, False, a float value, or 'log'")      

        # Initialize the Holt-Winters model hyperparameters
        self.seasonality = seasonality
        self.trend = trend
        self.seasonal_period = seasonal_period
        self.damped_trend = damped_trend
        self.initialization_method = initialization_method
        self.initial_level = initial_level
        self.initial_trend = initial_trend
        self.initial_seasonal = initial_seasonal
        self.use_boxcox = use_boxcox
        self.bounds = bounds
        self.dates = dates
        self.freq = freq
        self.missing = missing


    def fit(self, time_series, remove_bias=False):
            """
            Fits a Holt-Winters model to the given time series data.

            Parameters:
                time_series (np.ndarray): The time series data to fit the model to.
                remove_bias (bool, optional): Whether to remove the bias from the fitted model. Defaults to False.

            Raises:
                ValueError: If time_series is not a numpy array or if its length is less than 2 times the seasonal period.

            Returns:
                None
            """
            
            # Perform data validation
            if not isinstance(time_series, np.ndarray):
                raise ValueError("time_series must be a numpy array")
            if len(time_series) < 2 * self.seasonal_period:
                raise ValueError("time_series length must be greater than or equal to 2 * seasonal_period")

            # Fit a Holt-Winters model to the data
            model = ExponentialSmoothing(time_series, seasonal_periods=self.seasonal_period, trend=self.trend, seasonal=self.seasonality)
            self.fitted_model = model.fit(remove_bias=remove_bias)

    def forecast(self, steps=1):
        # Perform data validation
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        

        # Check if the fitted model exists
        if not hasattr(self, 'fitted_model') or self.fitted_model is None:
            raise ValueError("No fitted model found. Please fit the model before making predictions.")
        

        # Use fitted model to make predictions
        return self.fitted_model.forecast(steps)