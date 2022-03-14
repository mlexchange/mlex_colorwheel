import imageio
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft2, ifft2, fftshift
import PIL
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
from skimage.color import hsv2rgb    
from skimage import io
from cupy_common import check_cupy_available

gpu_accelerated = check_cupy_available()

if gpu_accelerated:
    print("Running on GPU")
    cp = __import__("cupy")
else:
    print("Running on CPU")
    cp = __import__("numpy")

#creates the colohweel
def bldclrwhl(nx, ny, sym):
    cda = cp.ones((nx, ny,2))
    cx = cp.linspace(-nx,nx,nx)
    cy = cp.linspace(-ny,ny,ny)
    cxx, cyy = cp.meshgrid(cy,cx)
    czz =(((cp.arctan2(cxx, cyy) / math.pi) + 1.0) / 2.0)*sym
    cd2 = cp.dstack((czz, cda))
    carr = cd2
    chi = cp.floor(carr[..., 0] * 6)
    f = carr[..., 0] * 6 - chi
    p = carr[..., 2] * (1 - carr[..., 1])
    q = carr[..., 2] * (1 - f * carr[..., 1])
    t = carr[..., 2] * (1 - (1 - f) * carr[..., 1])
    v = carr[..., 2]
    chi = cp.stack([chi, chi, chi], axis=-1).astype(cp.uint8) % 6
    out = cp.choose(
        chi, cp.stack([cp.stack((v, t, p), axis=-1),
                      cp.stack((q, v, p), axis=-1),
                      cp.stack((p, v, t), axis=-1),
                      cp.stack((p, q, v), axis=-1),
                      cp.stack((t, p, v), axis=-1),
                      cp.stack((v, p, q), axis=-1)]))

    if gpu_accelerated:
        imnp = cp.asnumpy(out)
    else:
        imnp = out
    return imnp

def nofft(whl, img, nx, ny):
    imnp = cp.array(img)
    fimg = cp.fft.fft2(imnp)
    whl  = cp.fft.fftshift(whl)
    proimg = cp.zeros((nx,ny,3))
    comb = cp.zeros((nx,ny,3), dtype=complex)
    magnitude = cp.repeat(np.abs(fimg)[:,:,np.newaxis], 3, axis=2)
    phase = cp.repeat(np.angle(fimg)[:,:,np.newaxis], 3, axis=2)
    proimg = whl*magnitude
    comb = cp.multiply(proimg, cp.exp(1j*phase))
    for n in range(3):
        proimg[:, :, n] = cp.real(cp.fft.ifft2(comb[:,:,n]))
        proimg[:, :, n] = proimg[:, :, n] - cp.min(proimg[:, :, n])
        proimg[:, :, n] = proimg[:, :, n] / cp.max(proimg[:, :, n])
    if gpu_accelerated:
        proimg = cp.asnumpy(proimg)        
    return proimg


#=========================================================================================
import os, io, time, base64, pathlib, zipfile, json
import pandas as pd
import dash
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import dash_uploader as du
#from jupyter_dash import JupyterDash
from PIL import Image
import plotly
import plotly.express as px

from plotly.data import iris

import holoviews as hv
from holoviews.plotting.plotly.dash import to_dash
from holoviews.operation.datashader import datashade

from urllib.parse import quote as urlquote
import time

# App Layout
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, "../assets/mlex-style.css"])

UPLOAD_FOLDER_ROOT = "../data/upload"
du.configure_upload(app, UPLOAD_FOLDER_ROOT, use_upload_id=False)
os.makedirs('.tmp', exist_ok=True)

def header():
    header = dbc.Navbar(
        dbc.Container([
            dbc.Row(
                [
                    dbc.Col(
                        html.Img(id="logo",
                                 src='assets/mlex.png',
                                 height="60px"),
                        md="auto"),
                    dbc.Col(
                        [html.Div(children=html.H3("MLExchange | Colorwheel Orientation"),
                                  id="app-title")],
                        md=True,
                        align="center",
                    )
                ],
                align="center",
            ),
            dbc.Row([
                dbc.Col([dbc.NavbarToggler(id="navbar-toggler")],
                        md=2)],
                align="center"),
        ],
        fluid=True),
        dark=True,
        color="dark",
        sticky="top",
    )
    return header

sidebar = dbc.Card(
    id='slidebar',
    children=[
        dbc.CardHeader(dbc.Label('Parameters', className='mr-2')),
        dbc.CardBody(
            children=[
                html.Div(children='''Symmetry'''),
                dcc.Slider(
                    id='symmetry-slider',
                    min=1,
                    max=12,
                    step=1,
                    value=6,
                    updatemode='mouseup',
                    marks={str(n): str(n) for n in range(13)}
                ),
                html.Div(children='''Color Saturation'''),
                dcc.Slider(
                    id='color-slider',
                    min=0,
                    max=20,
                    step=0.1,
                    value=1,
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode='mouseup'
                ), 
                html.Div(children='''Brightness'''),
                dcc.Slider(
                    id='bright-slider',
                    min=0,
                    max=4,
                    step=0.1,
                    value=1,
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode='mouseup'
                ), 
                html.Div(children='''Contrast'''),
                dcc.Slider(
                    id='contrast-slider',
                    min=0,
                    max=10,
                    step=0.1,
                    value=1,
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode='mouseup'
                ), 
                html.Div(children='''Blur'''),
                dcc.Slider(
                    id='blur-slider',
                    min=0,
                    max=10,
                    step=0.1,
                    value=0,
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode='mouseup'
                ),
                html.Div(children='''Overlap'''),
                dcc.Slider(
                    id='overlap',
                    min=0,
                    max=1,
                    step=0.1,
                    value=1.0,
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode='mouseup'
                )
            ]
        )
    ],
    style={'margin-left':'1rem'}
)

content = html.Div([
    du.Upload(
        id="dash-uploader",
        max_file_size=1800,  # 1800 Mb
        cancel_button=True,
        pause_button=True
    ),
    html.Div(id='graph', style={'margin-top': '1rem'}),
    html.Div(id='no-display',
         children=[
             dcc.Store(id='image-store', data={}),
             dcc.Store(id='cu_sym', data=-1),
             dcc.Store(id='temp-img', data={}),
             dcc.Store(id='new-img-flag', data=False),
             dcc.Store(id='list-filenames', data='')
         ])
],
style={'margin-top': '1rem', 'margin-right': '1rem'})



# Load iris dataset and replicate with noise to create large dataset
df_original = iris()[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
df = pd.concat([
df_original + np.random.randn(*df_original.shape) * 0.1
for i in range(10000)
])
dataset = hv.Dataset(df)

scatter = datashade(
hv.Scatter(dataset, kdims=["sepal_length"], vdims=["sepal_width"])
).opts(title="Datashader with %d points" % len(dataset))

components = to_dash(
app, [scatter], reset_button=True
)
#components.graphs[0]['id'] = {'type': 'graph', 'index': 0}
print(f'hv components kdims {components.kdims}')
datashader_display = html.Div(components.children)




app.layout = html.Div(children=[
                                header(),
                                datashader_display,
                                dbc.Row(children = [dbc.Col(sidebar, width=3), 
                                                    dbc.Col(content, width=9)],
                                       justify='center')
                                ])

# Returns the figure
def make_figure(image_npy, clrwhl=None):
    if clrwhl:
        height, width = np.array(image_npy).shape[0:2]
        fig = px.imshow(image_npy)
        fig.update_xaxes(
            showgrid=False,
            showticklabels=True, 
            zeroline=False,
            tickvals=np.linspace(start=0, stop=width, num=5),
            ticktext=np.linspace(start=225, stop=315, num=5)
        )
        fig.update_yaxes(
            showgrid=False,
            showticklabels=True, 
            zeroline=False,
            tickvals=np.linspace(start=0, stop=height, num=5),
            ticktext=np.linspace(start=135, stop=225, num=5)
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                          height=200)
    else:
        height, width = np.array(image_npy).shape[0:2]
        factor = int(np.sqrt(height*width/(700*700)))
        image = image_npy
        image = image_npy.resize((int(width/factor),int(height/factor)), Image.ANTIALIAS)
        fig = px.imshow(image)
        fig.update_xaxes(
            showgrid=False,
            showticklabels=False, 
            zeroline=False
        )
        fig.update_yaxes(
            showgrid=False,
            showticklabels=False, 
            zeroline=False
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                          height=300)
    return fig


def add_paths_from_dir(dir_path, supported_formats, list_file_path):
    '''
    Args:
        dir_path, str:            full path of a directory
        supported_formats, list:  supported formats, e.g., ['tiff', 'tif', 'jpg', 'jpeg', 'png']
        list_file_path, [str]:     list of absolute file paths
    
    Returns:
        Adding unique file paths to list_file_path, [str]
    '''
    root_path, list_dirs, filenames = next(os.walk(dir_path))
    for filename in filenames:
        exts = filename.split('.')
        if exts[-1] in supported_formats and exts[0] != '':
            file_path = root_path + '/' + filename
            if file_path not in list_file_path:
                list_file_path.append(file_path)
            
    for dirname in list_dirs:
        new_dir_path = dir_path + '/' + dirname
        list_file_path = add_paths_from_dir(new_dir_path, supported_formats, list_file_path)
    
    return list_file_path


@app.callback(
    [Output('new-img-flag', 'data'),
     Output('list-filenames', 'data'),
     Output('graph', 'children')],
    [Input('dash-uploader', 'isCompleted')],
    [State('dash-uploader', 'fileNames'),
     State('dash-uploader', 'upload_id')],
)
def image_upload(iscompleted, upload_filename, upload_id):
    if not iscompleted:
        return [False, '', []]
            
    list_filenames = []
    supported_formats = ['tiff', 'tif', 'jpg', 'jpeg', 'png']
    if upload_filename is not None:
        path_to_zip_file = pathlib.Path(UPLOAD_FOLDER_ROOT) / upload_filename[0]
        if upload_filename[0].split('.')[-1] == 'zip':   # unzip files and delete zip file
            zip_ref = zipfile.ZipFile(path_to_zip_file)  # create zipfile object
            path_to_folder = pathlib.Path(UPLOAD_FOLDER_ROOT) / upload_filename[0].split('.')[-2]
            if (upload_filename[0].split('.')[-2] + '/') in zip_ref.namelist():
                zip_ref.extractall(pathlib.Path(UPLOAD_FOLDER_ROOT))    # extract file to dir
            else:
                zip_ref.extractall(path_to_folder)
                
            zip_ref.close()  # close file
            os.remove(path_to_zip_file)
            list_filenames = add_paths_from_dir(str(path_to_folder), supported_formats, list_filenames)
        else:
            list_filenames.append(str(path_to_zip_file))

        image_slider_max = len(list_filenames)-1
    
    
    contents = [
        html.Div(id={'type': 'contents', 'index': 0},
             children=[html.H4(id={'type': 'graph-label', 'index': 0}),
                       dcc.Graph(id={'type': 'graph', 'index': 0},
                                 config={'displayModeBar': False}),
                       dcc.Slider(id={'type': 'image-slider', 'index': 0},
                             min=0,
                             max=image_slider_max,
                             value=0,
                             marks = {0: '0', image_slider_max: str(image_slider_max)},
                             updatemode='mouseup',
                             tooltip={"placement": "top", "always_visible": True})
                      ],
             style={'display': 'none'}
                ),
        
        html.Div(id={'type': 'contents', 'index':1},
             children=[dcc.Graph(id={'type': 'graph', 'index': 1},
                                 config={'displayModeBar': False},
                                 style={'margin-bottom': '1rem'}),
                       dbc.Button("SAVE", 
                                  id={'type':'save-data', 'index': 0}, 
                                  className="ms-auto", 
                                  n_clicks=0,
                                  style={'width': '95%'}),
                       dcc.Loading(id={'type':'download', 'index': 0}),
                      ],
             style={'display': 'none'}
                ),
                  
    ]

    return [True, list_filenames, contents]



# Define callback to update graph
@app.callback(
    [
        Output({'type': 'graph', 'index': ALL}, 'figure'),
        Output({'type': 'graph-label', 'index': ALL}, 'children'),
        Output({'type': 'contents', 'index': ALL}, 'style'),
        Output('cu_sym', 'data'),
    ],
    [
        Input('symmetry-slider', 'value'),
        Input('color-slider', 'value'),
        Input('bright-slider', 'value'), 
        Input('contrast-slider', 'value'), 
        Input('blur-slider', 'value'),
        Input('overlap', 'value'), 
        Input('new-img-flag', 'data'),
        Input('list-filenames', 'data'),
        Input({'type': 'image-slider', 'index': ALL}, 'value'),
        State('cu_sym', 'data')
    ]
)
def update_figure(symmetry, enh_val, bright_val, contra_val, blur_val, overlap, new_img_flag, list_filenames, 
                  slider_value, cu_sym):
    if new_img_flag is False:
        raise PreventUpdate
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    
    start = time.time()
    if 'overlap' in changed_id or 'new-img-flag' in changed_id or 'symmetry-slider' in changed_id or \
    'image-slider' in changed_id:
        im_label = list_filenames[slider_value[0]]
        original_image = np.array(Image.open(list_filenames[slider_value[0]]))
        if len(original_image.shape)>=3:
            original_image = original_image[:,:,:3]     #If 4 channels, convert to 3
            im = original_image[:,:,0]
        else:
            im = original_image
            original_image = original_image[:,:,np.newaxis]
        
        if 'overlap' not in changed_id :
            clrwhl = bldclrwhl(original_image.shape[0], original_image.shape[1],symmetry)
            cu_sym = symmetry
            graph2 = make_figure(clrwhl, True)
        else:
            clrwhl = np.load('.tmp/clrwhl.npy')
            graph2 = dash.no_update
        
        rgb = nofft(clrwhl, im, im.shape[0], im.shape[1])
        iim = np.repeat(im[:,:,np.newaxis], 3, axis=2)
        rgb2 = original_image * (1-overlap) + overlap * iim * rgb
        np.save('.tmp/rgb2.npy', rgb2)
        np.save('.tmp/clrwhl.npy', clrwhl)
    else:
        im_label = dash.no_update
        rgb2 = np.load('.tmp/rgb2.npy')
        graph2 = dash.no_update
    print('Colorwheel: ', time.time()-start)
    
    start = time.time()
    rgb2 = Image.fromarray(np.uint8(rgb2))
    img2 = rgb2.filter(ImageFilter.GaussianBlur(radius = blur_val)) 
    converter = PIL.ImageEnhance.Color(img2)
    img2 = converter.enhance(enh_val)
    converter = PIL.ImageEnhance.Brightness(img2)
    img2 = converter.enhance(bright_val)
    converter = PIL.ImageEnhance.Contrast(img2)
    img2 = converter.enhance(contra_val)
    img2.save('.tmp/colored_img.tif')
    print('Pillow: ', time.time()-start)
    
    start = time.time()
    graph1 = make_figure(img2)
    print('Make figure: ', time.time()-start)
    print('/n')
    return [graph1, graph2], [im_label], [{'width': '59%', 'display': 'inline-block', 'padding': '0 20'}, 
                                          {'display': 'inline-block', 'width': '39%', 'vertical-align': 'top',
                                           'margin-top': '5rem'}], cu_sym


@app.callback(
    Output({'type': 'download', 'index': ALL}, 'children'),
    Input({'type': 'save-data', 'index': ALL}, 'n_clicks'),
    State('list-filenames', 'data'),
    State({'type': 'image-slider', 'index': ALL}, 'value'),
    prevent_initial_call=True,
)
def func(n_clicks, list_filenames, slider_value):
    if any(n_clicks)>0:
        filename = list_filenames[slider_value[0]]
        if len(filename.split('.')) == 2:
            filename = 'colored_' + filename
        else:
            filename = 'colored_' + filename + '.tiff'
        image = Image.open('.tmp/colored_img.tif')
        url = "/download/" + urlquote(filename)
        return [html.A(download=filename, href=image, children=["Click here to start download"])]
    pass
                                
# Run app and display result inline in the notebook
# app.run_server(mode='inline')
app.run_server(host='0.0.0.0', port=8061, debug=True)




