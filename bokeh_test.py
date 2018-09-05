"""
Tool to experiment with FFT analysis. Contains 2 sinusoids, 1 with modifieable parameters and 1 fixed. Can also add in
White noise and a DC offset to signal.

To Run:
    $ bokeh serve --show bokeh_test.py

"""

import bokeh.models
import bokeh.layouts
import bokeh.plotting
import bokeh.palettes
import bokeh.io
import bokeh.embed
import numpy as np
import math


def make_dataset(sin_params, t0=0, tf=0.5, nt=50, dc_offset=0.0, noise=0.0, phase_threshold=100):
    """
    Creates a bokeh.models.ColumnDataSource with multiple (len(sin_params)) sinusoidal waves added together. May also
    include random (numpy.random.normal) noise of a constant amplitude.
    :param sin_params: Array of parameters for sinusoids. Array contains dicts with keys 'amp', 'freq', and 'phi' which
        specify the amplitude, frequency and phase of the sinusoid
    :param t0: Initial time for sinusoid
    :param tf: Final time for sinusoid
    :param nt: number of points in time series
    :param dc_offset: offset value for entire signal from zero
    :param noise: Amplitude of random noise to be added to sinusoid
    :param phase_threshold: Threshold value to elimiate having phase angles show up for values with low magnitude. When
        magnitude @ freq <  max(magnitude)/phase_threshold, will set the phase angle to zero.
    :return: bokeh.models.ColumnDataSource containing keys 't' (time) and 'y' (sinusoid) for plot
    """
    t = np.linspace(t0,tf,nt)
    y = np.zeros(t.shape)
    data = {'t': t}
    for i, sp in enumerate(sin_params):
        tmp_sin = sp['amp']*np.cos(2*math.pi*sp['freq']*t + sp['phi']*math.pi/180.0)
        y += tmp_sin
        data['y{i}'.format(i=i, A=sp['amp'], f=sp['freq'], phi=sp['phi']).format(i)] = tmp_sin # ={A}sin(2\pi{f}t+{phi})
    tmp_noise = noise*np.random.normal(size=t.size)
    data['White noise'.format(noise)] = tmp_noise

    y += tmp_noise
    y += dc_offset
    data['y'] = y

    spectrum = np.fft.rfft(y)
    freq = np.fft.rfftfreq(t.size, d=(t[1]-t[0]))
    magnitude = np.abs(spectrum)/nt
    # Filter out phase angles for very low magnitude points
    threshold = max(magnitude)/phase_threshold
    spectrum2 = spectrum
    spectrum2[magnitude < threshold] = 0.0
    phase = np.angle(spectrum2)*180.0/math.pi

    data_freq = {
        'freq': freq,
        'magnitude': magnitude,
        'phase': phase,
    }
    # data['color'] = bokeh.palettes.Accent8
    return bokeh.models.ColumnDataSource(data=data), bokeh.models.ColumnDataSource(data_freq)


def make_plot(src, src_freq):
    p = bokeh.plotting.figure()
    p_fourier_mag = bokeh.plotting.figure()
    p_fourier_phase = bokeh.plotting.figure(y_range=(-200,200))

    light_alpha = 1.0
    for ii, key in enumerate(src.data.keys()):
        color_plt = bokeh.palettes.Accent8[ii]
        print(color_plt)
        if key == 'y':
            # p.scatter(source=src, x='t', y=key, color='black')
            p.line(source=src, x='t', y=key, line_width=5, legend=dict(value=key), line_color=color_plt)
        if key == 't':
            pass
        else:
            # p.scatter(source=src, x='t', y=key, fill_alpha=light_alpha, fill_color=color_plt, legend=dict(value=key))
            p.line(source=src, x='t', y=key, line_alpha=light_alpha, line_color=color_plt, legend=dict(value=key))
    p.xaxis.axis_label = "Time [sec.]"
    p.yaxis.axis_label = "Signal Amplitude [Amplitude]"

    # Frequency Plot
    p_fourier_mag.vbar(source=src_freq, x='freq', top='magnitude', width=0.5)
    p_fourier_mag.x_range.start = freq_min
    p_fourier_mag.x_range.end = freq_max
    p_fourier_mag.xaxis.axis_label="Frequency [Hz]"
    p_fourier_mag.yaxis.axis_label="FFT Coefficient [Amplitude/2, Amplitude@f=0Hz]"

    p_fourier_phase.vbar(source=src_freq, x='freq', top='phase', width=0.5)
    p_fourier_phase.x_range.start = freq_min
    p_fourier_phase.x_range.end = freq_max
    p_fourier_phase.xaxis.axis_label = "Frequency [Hz]"
    p_fourier_phase.yaxis.axis_label = "FFT Phase from Cosine @ t=0 [degree]"

    return p, p_fourier_mag, p_fourier_phase


def update(attr, old, new):
    print('update')
    sin_params[0]['amp'] = float(text_amp[0].value)
    sin_params[0]['freq'] = slider_f[0].value
    sin_params[0]['phi'] = slider_phi[0].value
    # for ii, sin_param in sin_params:
    #     sin_params[ii]['amp'] = float(text_amp[ii].value)
    #     sin_params[ii]['freq'] = slider_f[ii].value
    #     sin_params[ii]['phi'] = slider_phi[ii].value
    new_src, new_src_freq = make_dataset(
        sin_params, noise=slider_noise.value, nt=nt, phase_threshold=slider_phase_threshold.value,
        dc_offset=dc_offset_slider.value)
    src.data.update(new_src.data)

    src_freq.data.update(new_src_freq.data)


# Initial Data
nt = 1000   # Number of points in original signal
sin_params = [{
        'amp': 1.0,
        'freq': 10.0,
        'phi': 90.0,
    },
    {
        'amp': 0.125,
        'freq': 60,
        'phi': 45.0,
    },
]
freq_min = 0
freq_max = 100

# Create Input Controls
text_amp = []
slider_f = []
slider_phi = []
for ii in range(1):
    text_amp.append(bokeh.models.widgets.TextInput(value=str(sin_params[ii]['amp']), title="A{}:".format(ii)))
    text_amp[ii].on_change('value', update)
    slider_f.append(bokeh.models.widgets.Slider(start=freq_min, end=freq_max, value=sin_params[ii]['freq'], step=2, title="freq_{}".format(ii)))
    slider_f[ii].on_change('value', update)
    slider_phi.append(bokeh.models.widgets.Slider(start=-180, end=180, value=sin_params[ii]['phi'], step=15, title="phi_{}".format(ii)))
    slider_phi[ii].on_change('value', update)

dc_offset_slider =bokeh.models.widgets.Slider(start=-3, end=3, value=0.0, step=0.1, title="DC Offset")
dc_offset_slider.on_change('value', update)
slider_noise = bokeh.models.widgets.Slider(start=0, end=0.25, value=0.00, step=0.01, title="Noise Amplitude")
slider_noise.on_change('value', update)
slider_phase_threshold = bokeh.models.widgets.Slider(start=1, end=1000, value=100, step=25, title="Phase/ Amplitude Cutoff")
slider_phase_threshold.on_change('value', update)

# Group Controls Together for Interface
sin_controls = bokeh.layouts.WidgetBox(dc_offset_slider, text_amp[0], slider_f[0], slider_phi[0],)
noise_controls = bokeh.layouts.WidgetBox(slider_noise, slider_phase_threshold,)
all_controls = bokeh.layouts.column(sin_controls, noise_controls)

src, src_freq = make_dataset(sin_params, noise=0.00, nt=nt)

# Make Plot
p, p_fourier_mag, p_fourier_phase = make_plot(src, src_freq)

# Put plot and controls together
layout = bokeh.layouts.row(all_controls, p, bokeh.layouts.column(p_fourier_mag, p_fourier_phase))
tab = bokeh.models.Panel(child=layout, title="Sum of Sinusoids")
tabs = bokeh.models.widgets.Tabs(tabs=[tab])

# Add it to the current document (displays plot)
bokeh.io.curdoc().add_root(tabs)
