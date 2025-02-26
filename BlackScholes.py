import unittest
from decimal import Decimal, getcontext
import numpy as np
from numpy import exp, sqrt, log
from scipy.stats import norm

# Set NumPy print precision
np.set_printoptions(precision=10, suppress=True)

class BlackScholes:
    def __init__(
        self,
        time_to_maturity: float,
        strike: float,
        current_price: float,
        volatility: float,
        interest_rate: float,
        call_purchase_price: float = None,
        put_purchase_price: float = None,
        purchase_time_to_maturity: float = None
    ):
        # Convert inputs to high precision
        self.time_to_maturity = float(np.format_float_positional(time_to_maturity, precision=10))
        self.strike = float(np.format_float_positional(strike, precision=10))
        self.current_price = float(np.format_float_positional(current_price, precision=10))
        self.volatility = float(np.format_float_positional(volatility, precision=10))
        self.interest_rate = float(np.format_float_positional(interest_rate, precision=10))
        self.call_purchase_price = float(np.format_float_positional(call_purchase_price, precision=10)) if call_purchase_price is not None else None
        self.put_purchase_price = float(np.format_float_positional(put_purchase_price, precision=10)) if put_purchase_price is not None else None
        self.purchase_time_to_maturity = float(np.format_float_positional(purchase_time_to_maturity or time_to_maturity, precision=10))

    def calculate_prices(self):
        # Use high precision for intermediate calculations
        d1 = (
            log(self.current_price / self.strike) +
            (self.interest_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity
        ) / (self.volatility * sqrt(self.time_to_maturity))
        
        d2 = d1 - self.volatility * sqrt(self.time_to_maturity)

        # Calculate prices with high precision
        call_price = float(np.format_float_positional(
            self.current_price * norm.cdf(d1) - 
            self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2),
            precision=10
        ))
        
        put_price = float(np.format_float_positional(
            self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-d2) -
            self.current_price * norm.cdf(-d1),
            precision=10
        ))

        self.call_price = call_price
        self.put_price = put_price

        # Calculate PnL with high precision
        if self.call_purchase_price is not None:
            if abs(self.time_to_maturity - self.purchase_time_to_maturity) < 1e-10 and \
               abs(self.current_price - self.current_price) < 1e-10:
                self.call_pnl = 0.0
                self.call_pnl_percentage = 0.0
            else:
                self.call_pnl = float(np.format_float_positional(
                    call_price - self.call_purchase_price, precision=10))
                self.call_pnl_percentage = float(np.format_float_positional(
                    (self.call_pnl / self.call_purchase_price * 100) if self.call_purchase_price != 0 else 0,
                    precision=10
                ))
        else:
            self.call_pnl = 0.0
            self.call_pnl_percentage = 0.0
        
        if self.put_purchase_price is not None:
            if abs(self.time_to_maturity - self.purchase_time_to_maturity) < 1e-10 and \
               abs(self.current_price - self.current_price) < 1e-10:
                self.put_pnl = 0.0
                self.put_pnl_percentage = 0.0
            else:
                self.put_pnl = float(np.format_float_positional(
                    put_price - self.put_purchase_price, precision=10))
                self.put_pnl_percentage = float(np.format_float_positional(
                    (self.put_pnl / self.put_purchase_price * 100) if self.put_purchase_price != 0 else 0,
                    precision=10
                ))
        else:
            self.put_pnl = 0.0
            self.put_pnl_percentage = 0.0

        # GREEKS
        self.call_delta = norm.cdf(d1)
        self.put_delta = -norm.cdf(-d1)
        self.gamma = norm.pdf(d1) / (self.current_price * self.volatility * sqrt(self.time_to_maturity))
        
        return call_price, put_price

    def calculate_risk_metrics(self):
        """Calculate various risk metrics for the options."""
        # Maximum possible losses (should be purchase price)
        self.max_call_loss = float(np.format_float_positional(
            abs(self.call_purchase_price) if self.call_purchase_price is not None else 0.0,
            precision=10
        ))
        self.max_put_loss = float(np.format_float_positional(
            abs(self.put_purchase_price) if self.put_purchase_price is not None else 0.0,
            precision=10
        ))
        
        # Maximum possible gains
        # For calls: theoretically unlimited
        if self.call_purchase_price is not None:
            self.max_call_gain = float('inf')  # Technically infinite
        else:
            self.max_call_gain = 0.0
            
        # For puts: maximum gain is strike price minus purchase price
        if self.put_purchase_price is not None:
            self.max_put_gain = float(np.format_float_positional(
                self.strike - self.put_purchase_price,
                precision=10
            ))
        else:
            self.max_put_gain = 0.0
        
        # Break-even points
        self.call_breakeven = float(np.format_float_positional(
            self.strike + (self.call_purchase_price or 0.0),
            precision=10
        ))
        self.put_breakeven = float(np.format_float_positional(
            self.strike - (self.put_purchase_price or 0.0),
            precision=10
        ))
        
        return {
            'max_call_loss': self.max_call_loss,
            'max_put_loss': self.max_put_loss,
            'max_call_gain': self.max_call_gain,
            'max_put_gain': self.max_put_gain,
            'call_breakeven': self.call_breakeven,
            'put_breakeven': self.put_breakeven
        }

class TestBlackScholes(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        # Test Case 1: At-the-money option
        self.bs_atm = BlackScholes(
            time_to_maturity=1.0,
            strike=100.0,
            current_price=100.0,
            volatility=0.2,
            interest_rate=0.05,
            call_purchase_price=8.0,
            put_purchase_price=6.0
        )
        
        # Test Case 2: Deep in-the-money call, out-of-the-money put
        self.bs_itm_call = BlackScholes(
            time_to_maturity=1.0,
            strike=90.0,
            current_price=100.0,
            volatility=0.2,
            interest_rate=0.05,
            call_purchase_price=15.0,
            put_purchase_price=2.0
        )
        
        # Test Case 3: Deep out-of-the-money call, in-the-money put
        self.bs_otm_call = BlackScholes(
            time_to_maturity=1.0,
            strike=110.0,
            current_price=100.0,
            volatility=0.2,
            interest_rate=0.05,
            call_purchase_price=3.0,
            put_purchase_price=10.0
        )

    def test_no_arbitrage(self):
        """Test put-call parity"""
        for bs in [self.bs_atm, self.bs_itm_call, self.bs_otm_call]:
            call_price, put_price = bs.calculate_prices()
            # Put-call parity: C - P = S - K*e^(-rt)
            left_side = call_price - put_price
            right_side = bs.current_price - bs.strike * exp(-bs.interest_rate * bs.time_to_maturity)
            self.assertAlmostEqual(left_side, right_side, places=7)

    def test_risk_metrics(self):
        """Test risk metrics calculations"""
        for bs in [self.bs_atm, self.bs_itm_call, self.bs_otm_call]:
            risk_metrics = bs.calculate_risk_metrics()
            
            # Test that max losses equal purchase prices
            self.assertAlmostEqual(risk_metrics['max_call_loss'], abs(bs.call_purchase_price), places=7)
            self.assertAlmostEqual(risk_metrics['max_put_loss'], abs(bs.put_purchase_price), places=7)
            
            # Test break-even points
            self.assertAlmostEqual(risk_metrics['call_breakeven'], bs.strike + bs.call_purchase_price, places=7)
            self.assertAlmostEqual(risk_metrics['put_breakeven'], bs.strike - bs.put_purchase_price, places=7)

if __name__ == '__main__':
    unittest.main()