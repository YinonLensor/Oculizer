import numpy as np
import random
import time
from oculizer.config import audio_parameters
from oculizer.light.effects import apply_effect

SAMPLERATE = audio_parameters['SAMPLERATE']
BLOCKSIZE = audio_parameters['BLOCKSIZE']

COLORS = np.array([
    [255, 0, 0],    # red
    [255, 127, 0],  # orange
    [255, 255, 0],  # yellow
    [0, 255, 0],    # green
    [0, 0, 255],    # blue
    [75, 0, 130],   # purple
    [255, 0, 255],  # pink
    [255, 255, 255], # white
    [0, 0, 0]       # black
], dtype=np.int32)

COLOR_NAMES = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'white']

# PANEL_COLORS = {
#     'red': 3,
#     'green': 6,
#     'blue': 9,
#     'yellow': 12,
#     'pink': 15,
#     'light_blue': 18,
#     'white': 21
# }

def scale_mfft(mfft_vec, scaling_factor=10.0, scaling_method='log'):
    num_bins = len(mfft_vec)
    if scaling_method == 'log':
        scaling = np.log1p(np.arange(num_bins) / num_bins) * scaling_factor + 1
    elif scaling_method == 'exp':
        scaling = np.exp(np.arange(num_bins) / num_bins * scaling_factor)
    elif scaling_method == 'linear':
        scaling = np.linspace(1, 1 + scaling_factor, num_bins)
    else:
        raise ValueError("Invalid scaling method. Choose 'log', 'exp', or 'linear'.")
    scaling /= scaling.mean()
    return mfft_vec * scaling

def color_to_rgb(color_name):
    return COLORS[COLOR_NAMES.index(color_name)] if color_name in COLOR_NAMES else COLORS[np.random.randint(0, len(COLOR_NAMES))]

def freq_to_index(freq):
    return int(freq * BLOCKSIZE / SAMPLERATE)

def power_to_brightness(power, power_range, brightness_range):
    power_low, power_high = power_range
    brightness_low, brightness_high = brightness_range
    if power <= power_low:
        return brightness_low
    elif power >= power_high:
        return brightness_high
    else:
        return int((power - power_low) / (power_high - power_low) * (brightness_high - brightness_low) + brightness_low)

def mfft_to_value(mfft_vec, mfft_range, power_range, value_range):
    mfft_low, mfft_high = int(mfft_range[0]), int(mfft_range[1])
    if mfft_low < 0 or mfft_high > len(mfft_vec):
        raise ValueError(f"MFFT range {mfft_range} is out of bounds for MFFT vector of length {len(mfft_vec)}")
    mfft_mean = np.mean(mfft_vec[mfft_low:mfft_high])
    return power_to_brightness(mfft_mean, power_range, value_range)

# --- Party light (7ch multi-effect: wash + pattern roller + laser + strobe) ---
# IMPORTANT: most channels are *quantized selectors*, not continuous dimmers.
# Fixture channel layout (1-indexed on the fixture itself):
#   1 strobe (ALL elements). Steps of 27: 0-26 off, then every +27 doubles the
#     rate up to 255. Strobe only flashes whatever is already lit.
#   2 wash LED colour (2 lamps, no pattern). Steps of 16: 0-15 off, then each
#     +16 is the next colour: red, green, blue, white, yellow, magenta, pink...
#   3 pattern-roller LEDs. 0-79 a single LED, 80+ a growing subset, 255 all on.
#   4 pattern-roller rotation. BOOL: 0-25 still, >25 spins at constant speed.
#   5 laser colour. Quantized: off / red / green / both.
#   6 laser-roller rotation. BOOL: 0-25 still, >25 spins at constant speed.
#   7 auto mode (>25 runs built-in presets). KEPT 0 — using it loses control.
#
# Scene fields: color (name|"random"|int), pattern (off|single|some|all|"audio"|
# "random"|int), pattern_spin (on/off/"audio"/int), laser_color (off|red|green|
# both|random), laser_spin (on/off/"audio"/int), strobe (off|slow|medium|fast|max|
# "beat"|"audio"|"random"|int) with strobe_threshold / strobe_level for "beat".

PARTY_STROBE_STEP = 27
PARTY_STROBE_NAMED = {'off': 0, 'slow': 27, 'medium': 81, 'fast': 162, 'max': 243}
PARTY_COLOR_STEP = 16
PARTY_COLOR_NAMES = ['off', 'red', 'green', 'blue', 'white', 'yellow', 'magenta', 'pink']
PARTY_LASER_COLORS = {'off': 0, 'red': 96, 'green': 160, 'both': 224}
PARTY_PATTERN_NAMED = {'off': 0, 'single': 40, 'some': 140, 'all': 255}
PARTY_SPIN_ON = 128  # any value >25 spins; 128 = clearly "on"


def _norm(power, power_range):
    lo, hi = power_range[0], power_range[1]
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (power - lo) / (hi - lo)))


def _party_color(val):
    if isinstance(val, str):
        if val == 'random':
            return int(np.random.randint(1, len(PARTY_COLOR_NAMES))) * PARTY_COLOR_STEP + 8
        idx = PARTY_COLOR_NAMES.index(val) if val in PARTY_COLOR_NAMES else 0
        return idx * PARTY_COLOR_STEP + (8 if idx > 0 else 0)
    return int(val)


def _party_laser(val):
    if isinstance(val, str):
        if val == 'random':
            return random.choice(list(PARTY_LASER_COLORS.values()))
        return PARTY_LASER_COLORS.get(val, 0)
    return int(val)


def _party_pattern(light, power=0.0, power_range=(0, 1)):
    p = light.get('pattern', 0)
    if isinstance(p, str):
        if p == 'random':
            return int(np.random.randint(0, 256))
        if p == 'audio':
            return power_to_brightness(power, power_range, [0, 255])
        return PARTY_PATTERN_NAMED.get(p, 0)
    return int(p)


def _party_spin(val, power=0.0, power_range=(0, 1)):
    """CH4/CH6 are on/off (>25 spins). Return 0 (still) or PARTY_SPIN_ON."""
    if isinstance(val, str):
        if val == 'audio':
            return PARTY_SPIN_ON if _norm(power, power_range) > 0.5 else 0
        return PARTY_SPIN_ON if val in ('on', 'spin', 'true', 'yes') else 0
    if isinstance(val, bool):
        return PARTY_SPIN_ON if val else 0
    return int(val)


def _party_strobe(light, power=0.0, power_range=(0, 1)):
    s = light.get('strobe', 0)
    if isinstance(s, str):
        if s == 'beat':
            threshold = light.get('strobe_threshold', power_range[1])
            on = PARTY_STROBE_NAMED.get(light.get('strobe_level', 'fast'), 162)
            return on if power >= threshold else 0
        if s == 'audio':
            band = int(_norm(power, power_range) * 9)  # 0..9 strobe steps
            return min(255, band * PARTY_STROBE_STEP)
        if s == 'random':
            return int(np.random.randint(0, 10)) * PARTY_STROBE_STEP
        return PARTY_STROBE_NAMED.get(s, 0)
    return int(s)


def _party_channels(strobe=0, color=0, pattern=0, pattern_spin=0, laser=0, laser_spin=0):
    # [strobe, wash colour, pattern LEDs, pattern spin, laser colour, laser spin, auto=0]
    return [int(strobe), int(color), int(pattern), int(pattern_spin),
            int(laser), int(laser_spin), 0]


def _fixture_len(light_type):
    return {'rockville864': 39, 'partylight': 7, 'discoball': 9}.get(light_type, 6)


def time_function(t, frequency, function):
    functions = {
        'sine': lambda t, f: np.sin(t * f * 2 * np.pi) * 0.5 + 0.5,
        'cosine': lambda t, f: np.cos(t * f * 2 * np.pi) * 0.5 + 0.5,
        'square': lambda t, f: np.sign(np.sin(t * f * 2 * np.pi)) * 0.5 + 0.5,
        'triangle': lambda t, f: np.abs(((t * f) % 2) - 1),
        'sawtooth_forward': lambda t, f: (t * f) % 1,
        'sawtooth_backward': lambda t, f: 1 - (t * f % 1)
    }
    return functions.get(function, functions['sine'])(t, frequency)

def process_mfft(light, mfft_vec):
    mfft_range = light.get('mfft_range', (0, len(mfft_vec)))
    power_range = light.get('power_range', (0, 1))
    value_range = light.get('brightness_range', (0, 255))
    
    if light['type'] == 'dimmer':
        return [mfft_to_value(mfft_vec, mfft_range, power_range, value_range)]
    
    elif light['type'] == 'rgb':
        brightness = mfft_to_value(mfft_vec, mfft_range, power_range, value_range)
        color = color_to_rgb(light.get('color', 'random'))
        strobe = light.get('strobe', 0)
        return [brightness, *color, strobe, 0]
    
    elif light['type'] == 'pinspot':
        brightness = mfft_to_value(mfft_vec, mfft_range, power_range, value_range)
        color = color_to_rgb(light.get('color', 'random'))
        strobe = light.get('strobe', 0)
        # Channel config: [R, G, B, W, MASTER_DIMMER, STROBE]
        # Scale color by brightness and keep W at 0
        scaled_color = [int(c * brightness / 255) for c in color]
        return [*scaled_color, 0, brightness, strobe]

    elif light['type'] == 'partylight':
        lo, hi = int(mfft_range[0]), int(mfft_range[1])
        power = float(np.mean(mfft_vec[lo:hi]))
        # Wash colour is a discrete pick; pattern LEDs fill in with audio by default.
        color_cfg = light.get('color', 'off')
        color = _party_color(color_cfg) if isinstance(color_cfg, str) else int(color_cfg)
        return _party_channels(
            strobe=_party_strobe(light, power, power_range),
            color=color,
            pattern=_party_pattern(light, power, power_range),
            pattern_spin=_party_spin(light.get('pattern_spin', light.get('rotation', 0)), power, power_range),
            laser=_party_laser(light.get('laser_color', 'off')),
            laser_spin=_party_spin(light.get('laser_spin', 0), power, power_range),
        )

    elif light['type'] == 'discoball':
        # 9ch: [master dimmer, strobe, R, G, B, W, auto, auto, auto]
        brightness = mfft_to_value(mfft_vec, mfft_range, power_range, value_range)
        r, g, b = color_to_rgb(light.get('color', 'random'))
        strobe = int(light.get('strobe', 0))
        return [brightness, strobe, int(r), int(g), int(b), 0, 0, 0, 0]

    elif light['type'] == 'strobe':
        threshold = light.get('threshold', 0.5)
        mfft_mean = np.mean(mfft_vec[mfft_range[0]:mfft_range[1]])
        return [255, 255] if mfft_mean >= threshold else [0, 0]
    
    elif light['type'] == 'laser':
        zoom_range = light.get('zoom_range', [0, 127])
        speed_range = light.get('speed_range', [0, 255])
        mfft_power = np.mean(mfft_vec)
        channels = [0] * 10
        channels[0] = 0
        if mfft_power <= power_range[0]:
            channels[1], channels[3], channels[9] = 0, zoom_range[0], speed_range[0]
        else:
            channels[1], channels[2] = 255, np.random.randint(0, 256)
            if mfft_power >= power_range[1]:
                channels[3], channels[9] = zoom_range[1], speed_range[1]
            else:
                power_ratio = (mfft_power - power_range[0]) / (power_range[1] - power_range[0])
                channels[3] = int(zoom_range[0] + power_ratio * (zoom_range[1] - zoom_range[0]))
                channels[9] = int(speed_range[0] + power_ratio * (speed_range[1] - speed_range[0]))
        return channels

    elif light['type'] == 'rockville864':
        try:
            # Handle panel component (8 sets of RGB bulbs)
            panel_config = light.get('panel', {})
            if panel_config.get('affect_panel', True):
                panel_mfft_range = panel_config.get('mfft_range', mfft_range)
                panel_power_range = panel_config.get('power_range', power_range)
                panel_value_range = panel_config.get('brightness_range', value_range)
                
                # Calculate panel brightness from audio
                brightness_magnitude = mfft_to_value(mfft_vec, panel_mfft_range, panel_power_range, panel_value_range)
                
                # Initialize 39 channels
                channels = [0] * 39
                
                # Channels 1-4: Master and panel effects
                channels[0] = 255  # Master dimmer always at max
                channels[1] = np.random.randint(0, 256) if panel_config.get('strobe') == 'random' else panel_config.get('strobe', 0)  # Panel strobe speed
                channels[2] = np.random.randint(126, 255) if panel_config.get('mode') == 'random' else panel_config.get('mode', 0)
                channels[3] = brightness_magnitude if panel_config.get('mode_speed', 255) == 'auto' else panel_config.get('mode_speed', 0)  # Mode speed
                
                # If mode is 0, use direct RGB control, otherwise RGB channels are background
                if channels[2] == 0:
                    color = color_to_rgb(panel_config.get('color', 'random'))
                    scaled_color = [int(c * brightness_magnitude / 255) for c in color]
                    
                    # Set all 8 RGB bulb sets (channels 5-28)
                    for i in range(8):
                        base_idx = 4 + (i * 3)
                        channels[base_idx:base_idx + 3] = scaled_color
                else:
                    # When mode is active, use color as background
                    color = color_to_rgb(panel_config.get('color', 'random'))
                    scaled_color = [int(c * brightness_magnitude / 255) for c in color]
                    # Set background color for all RGB channels
                    for i in range(8):
                        base_idx = 4 + (i * 3)
                        channels[base_idx:base_idx + 3] = scaled_color
            else:
                channels = [0] * 39
            
            bar_config = light.get('bar', {})
            if bar_config.get('affect_bar', True):
                bar_mfft_range = bar_config.get('bar_mfft_range', (0, len(mfft_vec)))
                bar_power = np.mean(mfft_vec[bar_mfft_range[0]:bar_mfft_range[1]])
                bar_threshold = bar_config.get('threshold', 0.5)
                bar_mode = bar_config.get('mode', 0)

                if bar_power >= bar_threshold:
                    channels[28] = bar_config.get('strobe', 0)
                    channels[29] = 0 if bar_mode == 'random' else bar_mode
                    channels[30] = bar_config.get('mode_speed', 255)
            
                    if bar_mode == 0:   
                        for i in range(31, 39):
                            channels[i] = random.choice(bar_config.get('bar_colors', [0, 255]))
                    elif bar_mode == 'random':
                        for i in range(31, 39):
                            channels[i] = random.choice([0, 255])
                    else:
                        for i in range(31, 39):
                            channels[i] = 0
                else:
                    # If bar is not affected, ensure all bar channels are off
                    for i in range(28, 39):
                        channels[i] = 0

            return channels
                    
        except Exception as e:
            print(f"Error processing rockville864 light {light['name']}: {e}")
            return [0] * 39

def process_bool(light):
    if light['type'] == 'rockville864':
        try:
            # Initialize 39 channels
            channels = [0] * 39
            channels[0] = 255  # Master dimmer always at max
            
            # Handle panel section
            panel_config = light.get('panel', {})
            
            # Set panel controls
            channels[1] = np.random.randint(0, 256) if panel_config.get('strobe') == 'random' else panel_config.get('strobe', 0)
            channels[2] = panel_config.get('mode', 0)
            channels[3] = panel_config.get('mode_speed', 255)
            
            # Handle panel brightness
            if panel_config.get('brightness') == 'random':
                min_brightness = panel_config.get('min_brightness', 0)
                max_brightness = panel_config.get('max_brightness', 255)
                brightness = np.random.randint(min_brightness, max_brightness + 1)
            else:
                brightness = panel_config.get('brightness', 255)
            
            # If mode is 0, use direct RGB control
            if channels[2] == 0:
                color = color_to_rgb(panel_config.get('color', 'random'))
                scaled_color = [int(c * brightness / 255) for c in color]
                
                # Set all 8 RGB bulb sets
                for i in range(8):
                    base_idx = 4 + (i * 3)
                    channels[base_idx:base_idx + 3] = scaled_color
            else:
                # When mode is active, use color as background
                color = color_to_rgb(panel_config.get('color', 'random'))
                scaled_color = [int(c * brightness / 255) for c in color]
                # Set background color for all RGB channels
                for i in range(8):
                    base_idx = 4 + (i * 3)
                    channels[base_idx:base_idx + 3] = scaled_color
            
            # Handle bar section
            bar_config = light.get('bar', {})
            if bar_config.get('affect_bar', True):
                channels[28] = bar_config.get('strobe', 0)
                channels[29] = 0 if bar_config.get('mode') == 'random' else bar_config.get('mode', 0)
                channels[30] = bar_config.get('mode_speed', 255)
                brightness = bar_config.get('brightness', 0)  # Default to 0 for bar
                
                if channels[29] == 0:  # Manual mode
                    channels[31:39] = [brightness] * 8
                elif channels[29] == "random":
                    channels[29] = 0  # Set to manual mode
                    # Generate 8 different random values
                    min_brightness = bar_config.get('min_brightness', 0)
                    max_brightness = bar_config.get('max_brightness', 255)
                    channels[31:39] = [np.random.randint(min_brightness, max_brightness + 1) for _ in range(8)]
                else:
                    channels[31:39] = [0] * 8
            else:
                # If affect_bar is False, set all bar channels to 0
                channels[28:39] = [0] * 11  
            
            return channels
            
        except Exception as e:
            print(f"Error processing rockville864 light {light['name']}: {e}")
            return [0] * 39
            
    elif light['type'] == 'dimmer':
        if light.get('brightness', 'random') == 'random':
            brightness = np.random.randint(light.get('min_brightness', 0), light.get('max_brightness', 255) + 1)
        else:
            brightness = light.get('brightness', 255)
        return [brightness]
    elif light['type'] == 'rgb':
        if light.get('brightness', 'random') == 'random':
            min_brightness = light.get('min_brightness', 0)
            max_brightness = light.get('max_brightness', 255)
            brightness = np.random.randint(min_brightness, max_brightness + 1)
        else:
            brightness = light.get('brightness', 255)
        color = color_to_rgb(light.get('color', 'random'))
        strobe = np.random.randint(0, 256) if light.get('strobe', 'random') == 'random' else light.get('strobe', 0)
        colorfade = light.get('colorfade', 0)
        return [brightness, *color, strobe, colorfade]
    elif light['type'] == 'pinspot':
        if light.get('brightness', 'random') == 'random':
            min_brightness = light.get('min_brightness', 0)
            max_brightness = light.get('max_brightness', 255)
            brightness = np.random.randint(min_brightness, max_brightness + 1)
        else:
            brightness = light.get('brightness', 255)
        color = color_to_rgb(light.get('color', 'random'))
        strobe = np.random.randint(0, 256) if light.get('strobe', 'random') == 'random' else light.get('strobe', 0)
        # Channel config: [R, G, B, W, MASTER_DIMMER, STROBE]
        # Scale color by brightness and keep W at 0
        scaled_color = [int(c * brightness / 255) for c in color]
        return [*scaled_color, 0, brightness, strobe]
    elif light['type'] == 'partylight':
        # Static look (no audio): "beat"/"audio" fall back to off.
        strobe = light.get('strobe', 0)
        if strobe in ('beat', 'audio'):
            strobe = 0
        else:
            strobe = _party_strobe(light)
        return _party_channels(
            strobe=strobe,
            color=_party_color(light.get('color', 'off')),
            pattern=_party_pattern(light),
            pattern_spin=_party_spin(light.get('pattern_spin', light.get('rotation', 0))),
            laser=_party_laser(light.get('laser_color', 'off')),
            laser_spin=_party_spin(light.get('laser_spin', 0)),
        )
    elif light['type'] == 'discoball':
        if light.get('brightness', 'random') == 'random':
            brightness = np.random.randint(light.get('min_brightness', 0), light.get('max_brightness', 255) + 1)
        else:
            brightness = light.get('brightness', 255)
        r, g, b = color_to_rgb(light.get('color', 'random'))
        strobe = int(light.get('strobe', 0))
        return [int(brightness), strobe, int(r), int(g), int(b), 0, 0, 0, 0]
    elif light['type'] == 'strobe':
        speed = np.random.randint(0, 256) if light.get('speed', 'random') == 'random' else light.get('speed', 255)
        brightness = np.random.randint(0, 256) if light.get('brightness', 'random') == 'random' else light.get('brightness', 255)
        return [speed, brightness]
    elif light['type'] == 'laser':
        return [128, 255] + [0] * 8

def process_time(light, current_time):
    if light['type'] == 'rockville864':
        try:
            frequency = light.get('frequency', 1)
            function = light.get('function', 'sine')
            min_value = light.get('min_brightness', 0)
            max_value = light.get('max_brightness', 255)
            
            # Calculate time-based value
            value = int(min_value + (max_value - min_value) * time_function(current_time, frequency, function))
            
            # Initialize channels
            channels = [0] * 39
            channels[0] = 255  # Master dimmer always at max
            
            # Handle panel section
            panel_config = light.get('panel', {})
            target = panel_config.get('target', 'brightness')  # What the time function affects
            
            channels[1] = panel_config.get('strobe', 0)
            channels[2] = panel_config.get('mode', 0)
            
            if target == 'mode_speed':
                channels[3] = value
            else:
                channels[3] = panel_config.get('mode_speed', 255)
            
            # If mode is 0, use direct RGB control
            if channels[2] == 0:
                color = color_to_rgb(panel_config.get('color', 'random'))
                if target == 'brightness':
                    scaled_color = [int(c * value / 255) for c in color]
                else:
                    scaled_color = color
                    
                # Set all 8 RGB bulb sets
                for i in range(8):
                    base_idx = 4 + (i * 3)
                    channels[base_idx:base_idx + 3] = scaled_color
            else:
                # When mode is active, RGB channels become background
                channels[4:28] = [value if target == 'brightness' else panel_config.get('brightness', 255)] * 24
            
            # Handle bar section
            bar_config = light.get('bar', {})
            if bar_config.get('affect_bar', True):  # Only process bar if affect_bar is True
                bar_target = bar_config.get('target', 'none')
                
                # Set basic bar controls
                channels[28] = bar_config.get('strobe', 0)
                channels[29] = bar_config.get('mode', 0)
                channels[30] = bar_config.get('mode_speed', 255)  # Mode speed
                
                if channels[29] == 0:  # Manual mode
                    if bar_target == 'sections':
                        # Apply time function to all sections
                        channels[31:39] = [value] * 8
                    else:
                        # Use configured sections
                        sections = bar_config.get('sections', [255] * 8)
                        channels[31:39] = sections
                elif channels[29] == "random":
                    channels[29] = 0
                    channels[31:39] = [int(np.random.randint(0, 256) * value / 255) for _ in range(8)]
                else:
                    channels[30] = bar_config.get('mode_speed', 255)  # Set mode speed
                    channels[31:39] = [0] * 8
            else:
                # If affect_bar is False, set all bar channels to 0
                channels[28:] = [0] * 12
            
            return channels
            
        except Exception as e:
            print(f"Error processing rockville864 light {light['name']}: {e}")
            return [0] * 39
            
    elif light['type'] == 'dimmer':
        # get the time function value
        value = int(light.get('min_brightness', 0) + (light.get('max_brightness', 255) - light.get('min_brightness', 0)) * time_function(current_time, light.get('frequency', 1), light.get('function', 'sine')))
        return [value]

    elif light['type'] == 'rgb':
        color = color_to_rgb(light.get('color', 'random'))
        strobe = light.get('strobe', 0)
        # get the time function value
        value = int(light.get('min_brightness', 0) + (light.get('max_brightness', 255) - light.get('min_brightness', 0)) * time_function(current_time, light.get('frequency', 1), light.get('function', 'sine')))
        return [value, *color, strobe, 0]
    
    elif light['type'] == 'pinspot':
        color = color_to_rgb(light.get('color', 'random'))
        strobe = light.get('strobe', 0)
        # get the time function value
        value = int(light.get('min_brightness', 0) + (light.get('max_brightness', 255) - light.get('min_brightness', 0)) * time_function(current_time, light.get('frequency', 1), light.get('function', 'sine')))
        # Channel config: [R, G, B, W, MASTER_DIMMER, STROBE]
        # Scale color by brightness and keep W at 0
        scaled_color = [int(c * value / 255) for c in color]
        return [*scaled_color, 0, value, strobe]

    elif light['type'] == 'partylight':
        lo = light.get('min_value', light.get('min_brightness', 0))
        hi = light.get('max_value', light.get('max_brightness', 255))
        value = int(lo + (hi - lo) * time_function(current_time, light.get('frequency', 1), light.get('function', 'sine')))
        kw = dict(
            strobe=_party_strobe(light) if not isinstance(light.get('strobe', 0), str) or light.get('strobe') in PARTY_STROBE_NAMED else 0,
            color=_party_color(light.get('color', 'off')),
            pattern=_party_pattern(light),
            pattern_spin=_party_spin(light.get('pattern_spin', light.get('rotation', 0))),
            laser=_party_laser(light.get('laser_color', 'off')),
            laser_spin=_party_spin(light.get('laser_spin', 0)),
        )
        target = light.get('target', 'pattern')
        if target in kw:
            kw[target] = value
        return _party_channels(**kw)

    elif light['type'] == 'discoball':
        value = int(light.get('min_brightness', 0) + (light.get('max_brightness', 255) - light.get('min_brightness', 0))
                    * time_function(current_time, light.get('frequency', 1), light.get('function', 'sine')))
        r, g, b = color_to_rgb(light.get('color', 'random'))
        strobe = int(light.get('strobe', 0))
        return [value, strobe, int(r), int(g), int(b), 0, 0, 0, 0]

    elif light['type'] == 'strobe':
        target = light.get('target', 'both')
        speed_range = light.get('speed_range', [0, 255])
        brightness_range = light.get('brightness_range', [0, 255])
        if target == 'speed':
            return [light.get('brightness', 255), brightness_range[1]]
        elif target == 'brightness':
            return [speed_range[1], light.get('brightness', 255)]
        else:
            speed = int(speed_range[0] + (speed_range[1] - speed_range[0]) * time_function(current_time, light.get('frequency', 1), light.get('function', 'sine')))
            brightness = int(brightness_range[0] + (brightness_range[1] - brightness_range[0]) * time_function(current_time, light.get('frequency', 1), light.get('function', 'sine')))
            return [speed, brightness]

def process_light(light, mfft_vec, current_time, modifiers=None):
    # First check if there's an effect
    if 'effect' in light:
        effect_config = light['effect']
        
        # Handle time modulation if configured
        if 'time_modulation' in effect_config:
            time_mod = effect_config['time_modulation']
            if not modifiers:
                modifiers = {}
            # Calculate time-based brightness scale
            brightness_scale = time_function(
                current_time,
                time_mod.get('frequency', 1.0),
                time_mod.get('function', 'sine')
            )
            # Scale between min and max
            min_bright = time_mod.get('min_brightness', 0) / 255
            max_bright = time_mod.get('max_brightness', 255) / 255
            brightness_scale = min_bright + (max_bright - min_bright) * brightness_scale
            modifiers['brightness_scale'] = brightness_scale
        
        if isinstance(effect_config, str):
            # Simple case: just effect name
            channels = [0] * _fixture_len(light['type'])  # Initialize channels
            channels = apply_effect(effect_config, channels, mfft_vec, {}, light['name'])
        elif isinstance(effect_config, dict):
            # Advanced case: effect name and config
            effect_name = effect_config.pop('name')
            channels = [0] * _fixture_len(light['type'])  # Initialize channels
            channels = apply_effect(effect_name, channels, mfft_vec, effect_config, light['name'])
            effect_config['name'] = effect_name  # Restore the name key
    else:
        # No effect, use modulator
        modulator = light.get('modulator', 'bool')
        
        # Get initial channel values based on modulator
        if modulator == 'mfft':
            channels = process_mfft(light, mfft_vec)
        elif modulator == 'bool':
            channels = process_bool(light)
        elif modulator == 'time':
            channels = process_time(light, current_time)
        else:
            return None
    
    # Apply modifiers if provided
    if channels and modifiers:
        if 'brightness_scale' in modifiers:
            # Scale brightness channels based on the modifier
            if light['type'] == 'dimmer':
                channels[0] = int(channels[0] * modifiers['brightness_scale'])
            elif light['type'] == 'rgb':
                channels[0] = int(channels[0] * modifiers['brightness_scale'])
            elif light['type'] == 'pinspot':
                # For pinspot, scale RGB channels and master dimmer
                channels[0] = int(channels[0] * modifiers['brightness_scale'])  # R
                channels[1] = int(channels[1] * modifiers['brightness_scale'])  # G
                channels[2] = int(channels[2] * modifiers['brightness_scale'])  # B
                channels[4] = int(channels[4] * modifiers['brightness_scale'])  # MASTER_DIMMER
            elif light['type'] == 'discoball':
                # 9ch [dimmer, strobe, R, G, B, W, ...]; scale the master dimmer
                channels[0] = int(channels[0] * modifiers['brightness_scale'])
            elif light['type'] == 'rockville864':
                # For Rockville, scale all RGB values
                for i in range(4, 28, 3):  # RGB sections
                    for j in range(3):  # R, G, B channels
                        channels[i + j] = int(channels[i + j] * modifiers['brightness_scale'])
    
    return channels

def main():
    mfft_data = np.random.rand(128)
    light = {
        'modulator': 'mfft',
        'type': 'rgb',
        'mfft_range': (0, 128),
        'power_range': (0, 1),
        'brightness_range': (0, 255),
        'color': 'random',
        'strobe': 0
    }
    dmx_values = process_light(light, mfft_data, time.time())
    print(dmx_values)

if __name__ == "__main__":
    main()



