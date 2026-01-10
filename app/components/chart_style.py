"""
Chart styling module for consistent, elegant Plotly charts.
"""

# Elegant color palette
COLORS = {
    'primary': '#1E3A5F',      # Navy blue
    'secondary': '#E07B53',    # Soft coral
    'accent': '#2E8B57',       # Sea green
    'positive': '#3498DB',     # Sky blue
    'negative': '#E74C3C',     # Soft red
    'neutral': '#7F8C8D',      # Gray
    'background': '#FFFFFF',
    'grid': '#E8E8E8',
    'text': '#2C3E50',
    'text_light': '#7F8C8D',
}

# Chart template
PLOTLY_TEMPLATE = {
    'layout': {
        'paper_bgcolor': COLORS['background'],
        'plot_bgcolor': COLORS['background'],
        'font': {
            'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            'size': 12,
            'color': COLORS['text'],
        },
        'title': {
            'font': {
                'size': 16,
                'color': COLORS['text'],
            },
            'x': 0.5,
            'xanchor': 'center',
        },
        'xaxis': {
            'gridcolor': COLORS['grid'],
            'gridwidth': 1,
            'linecolor': COLORS['grid'],
            'linewidth': 1,
            'tickfont': {'size': 11},
            'title': {'font': {'size': 12}},
            'zeroline': False,
        },
        'yaxis': {
            'gridcolor': COLORS['grid'],
            'gridwidth': 1,
            'linecolor': COLORS['grid'],
            'linewidth': 1,
            'tickfont': {'size': 11},
            'title': {'font': {'size': 12}},
            'zeroline': False,
        },
        'legend': {
            'bgcolor': 'rgba(255,255,255,0.9)',
            'bordercolor': COLORS['grid'],
            'borderwidth': 1,
            'font': {'size': 11},
        },
        'margin': {'l': 60, 'r': 30, 't': 50, 'b': 50},
        'hoverlabel': {
            'bgcolor': COLORS['background'],
            'bordercolor': COLORS['grid'],
            'font': {'size': 12, 'color': COLORS['text']},
        },
    }
}


def get_elegant_layout(**kwargs):
    """
    Get an elegant layout dictionary for Plotly charts.
    
    Args:
        **kwargs: Additional layout parameters to override defaults
        
    Returns:
        dict: Layout configuration
    """
    layout = {
        'paper_bgcolor': COLORS['background'],
        'plot_bgcolor': COLORS['background'],
        'font': {
            'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            'size': 12,
            'color': COLORS['text'],
        },
        'xaxis': {
            'gridcolor': COLORS['grid'],
            'gridwidth': 1,
            'linecolor': COLORS['grid'],
            'showgrid': True,
            'zeroline': False,
        },
        'yaxis': {
            'gridcolor': COLORS['grid'],
            'gridwidth': 1,
            'linecolor': COLORS['grid'],
            'showgrid': True,
            'zeroline': False,
        },
        'margin': {'l': 60, 'r': 30, 't': 50, 'b': 50},
        'hoverlabel': {
            'bgcolor': COLORS['background'],
            'font': {'size': 12},
        },
        'legend': {
            'bgcolor': 'rgba(255,255,255,0.9)',
            'bordercolor': COLORS['grid'],
            'borderwidth': 1,
        },
    }
    
    # Override with kwargs
    for key, value in kwargs.items():
        if isinstance(value, dict) and key in layout and isinstance(layout[key], dict):
            layout[key].update(value)
        else:
            layout[key] = value
    
    return layout


def style_timeseries_line():
    """Get styling for time series line."""
    return dict(color=COLORS['primary'], width=2)


def style_secondary_line():
    """Get styling for secondary line."""
    return dict(color=COLORS['secondary'], width=2)


def style_bar_colors(values):
    """
    Get colors for bar chart based on positive/negative values.
    
    Args:
        values: Array of values
        
    Returns:
        list: Colors for each bar
    """
    return [COLORS['positive'] if v >= 0 else COLORS['negative'] for v in values]


def get_hline_style(color='gray', dash='dash'):
    """Get styling for horizontal lines."""
    return dict(
        line_color=color,
        line_width=1,
        line_dash=dash,
    )
