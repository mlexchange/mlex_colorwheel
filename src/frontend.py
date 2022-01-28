#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
    imnp = cp.asnumpy(out)
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
            
    proimg = cp.asnumpy(proimg)
    return proimg

# In[10]:


import os, io, time, base64, pathlib, zipfile, json

import dash
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import dash_uploader as du
from flask_caching import Cache
from jupyter_dash import JupyterDash
from PIL import Image
import plotly
import plotly.express as px
from urllib.parse import quote as urlquote

# App Layout
app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, "../assets/mlex-style.css"])

UPLOAD_FOLDER_ROOT = "./upload"
du.configure_upload(app, UPLOAD_FOLDER_ROOT, use_upload_id=False)

def header():
    header= dbc.Navbar(
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
    ]
)

content = html.Div([
    du.Upload(
            id="dash-uploader",
            max_file_size=1800,  # 1800 Mb
            cancel_button=True,
            pause_button=True,
    ),
    html.Div(id='graph'),
    dcc.Loading(id='download'),
    html.Div(id='no-display',
             children=[
                 dcc.Store(id='image-store', data={}),
                 dcc.Store(id='cu_sym', data=-1),
                 dcc.Store(id='temp-img', data={}),
                 dcc.Store(id='clrwhl', data=[]),
                 dcc.Store(id='original_image', data=[]),
                 dcc.Store(id='rgb', data=[])
             ])
],
style={'margin-top': '1rem', 'margin-right': '1rem'})

app.layout = html.Div(children=[
                                header(),
                                dbc.Row([dbc.Col(sidebar, width=3), 
                                         dbc.Col(content, width=9)])
                                ],
                     style={'margin-top': '3rem',
                            'margin-bottom': '3rem',
                            'margin-left': '3rem',
                            'margin-right': '3rem'
                           })


# Returns the figure
def make_figure(image_npy, clrwhl=None):
    height, width = np.array(image_npy).shape[0:2]
    fig = px.imshow(image_npy)
    if clrwhl:
        fig.update_xaxes(
            showgrid=False,
            range=(0, width),
            showticklabels=True, 
            zeroline=False,
            tickvals=np.linspace(start=0, stop=width, num=5),
            ticktext=np.linspace(start=225, stop=315, num=5)
        )
        fig.update_yaxes(
            showgrid=False,
            range=(height, 0),
            showticklabels=True, 
            zeroline=False,
            tickvals=np.linspace(start=0, stop=height, num=5),
            ticktext=np.linspace(start=135, stop=225, num=5)
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                          height=200)
    else:
        fig.update_xaxes(
            showgrid=False,
            #range=(0, width),
            showticklabels=False, 
            zeroline=False
        )
        fig.update_yaxes(
            showgrid=False,
            #range=(height, 0),
            showticklabels=False, 
            zeroline=False
        ) 
    return fig


# Reads the image in cache and returns it as a numpy array
def read_img_cache(image_cache):
    # image_cache is a dict, keys=filename, value=bin encoding
    img_bytes = base64.b64decode(image_cache)
    im_bbytes = io.BytesIO(img_bytes)
    im = PIL.Image.open(im_bbytes)
    return np.array(im)


@app.callback(
    [Output('image-store', 'data'),
     Output('graph', 'children')],
    [Input('dash-uploader', 'isCompleted')],
    [State('dash-uploader', 'fileNames'),
     State('dash-uploader', 'upload_id')],
)
def image_upload(iscompleted, upload_filename, upload_id):
    if not iscompleted:
        return [{}, []]
    
    image_store_data = {}
    
#     if upload_filename is not None:
#         print(f'upload_filename {upload_filename}')
#         for filename in upload_filename:
#             file = pathlib.Path(UPLOAD_FOLDER_ROOT) / filename
#             image_store_data[filename] = np.array(PIL.Image.open(file))
    
    if upload_filename is not None:
        filename = upload_filename[0]
        if filename.split(".")[1] == 'zip':
            path_to_zip_file = pathlib.Path(UPLOAD_FOLDER_ROOT) / filename
            with zipfile.ZipFile(path_to_zip_file) as z:
                for fname in z.namelist():
                    if fname.split('.')[-1] in ['tiff', 'tif', 'jpg', 'jpeg', 'png']:
                        with z.open(fname) as f:
                            try:
                                image_store_data[fname.split("/")[-1]] = np.array(PIL.Image.open(f))
                            except:
                                continue
            
            os.remove(path_to_zip_file)
        else:
            path_to_file = pathlib.Path(UPLOAD_FOLDER_ROOT) / filename
            image_store_data[filename] = np.array(PIL.Image.open(path_to_file))
            os.remove(path_to_file)
        image_slider_max = len(image_store_data)-1
        
    contents = [html.Div(id={'type': 'contents', 'index': 0},
                         children=[dcc.Graph(id={'type': 'graph', 'index': 0},
                                             config={'displayModeBar': False}),
                                   dcc.Slider(id={'type': 'image-slider', 'index': 0},
                                         min=0,
                                         max=image_slider_max,
                                         value=0,
                                         marks = {0: '0', image_slider_max: str(image_slider_max)},
                                         updatemode='mouseup',
                                         tooltip={"placement": "top", "always_visible": True})],
                         style={'display': 'none'}),
                html.Div(id={'type': 'contents', 'index':1},
                         children=[dcc.Graph(id={'type': 'graph', 'index': 1},
                                             config={'displayModeBar': False},
                                             style={'margin-bottom': '1rem'}),
                                   dbc.Button("SAVE", 
                                              id={'type':'save-data', 'index': 0}, 
                                              className="ms-auto", 
                                              n_clicks=0,
                                              style={'width': '95%'})],
                         style={'display': 'none'}),
                
               ]

    return [image_store_data, contents]



# Define callback to update graph
@app.callback(
    [
        Output({'type': 'graph', 'index': ALL}, 'figure'),
        Output({'type': 'contents', 'index': ALL}, 'style'),
        Output('temp-img', 'data'),
        Output('clrwhl', 'data'),
        Output('cu_sym', 'data'),
    ],
    [
        Input('symmetry-slider', 'value'),
        Input('color-slider', 'value'),
        Input('bright-slider', 'value'), 
        Input('contrast-slider', 'value'), 
        Input('blur-slider', 'value'),
        Input('overlap', 'value'), 
        Input('image-store', 'data'),
        Input({'type': 'image-slider', 'index': ALL}, 'value'),
        State('cu_sym', 'data'),
        State('temp-img', 'data'),
        State('clrwhl', 'data')
    ]
)
def update_figure(symmetry, enh_val, bright_val, contra_val, blur_val, overlap, image_store_data, slider_value, 
                  cu_sym, rgb2, clrwhl):
    start = time.time()
    if len(image_store_data)==0:
        raise PreventUpdate
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    
    if 'overlap' in changed_id or 'image-store' in changed_id or 'symmetry-slider' in changed_id or     'image-slider' in changed_id:
        if slider_value[0] is not None:
            if slider_value[0]<len(image_store_data):
                im_cache = image_store_data[list(image_store_data.keys())[slider_value[0]]]
        else:
            im_cache = image_store_data[list(image_store_data.keys())[0]]
        
        original_image = np.array(im_cache)
        if len(original_image.shape)==3:
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
            clrwhl = np.array(json.loads(clrwhl), dtype='uint8')
            graph2 = dash.no_update
        
        rgb = nofft(clrwhl, im, im.shape[0], im.shape[1])
        iim = np.repeat(im[:,:,np.newaxis], 3, axis=2)
        rgb2 = original_image * (1-overlap) + overlap * iim * rgb
        clrwhl = json.dumps(clrwhl.tolist())
    else:
        content_type, content_string = rgb2.split(',')
        rgb2 = read_img_cache(content_string)
        graph2 = dash.no_update
    
    rgb2 = Image.fromarray(np.uint8(rgb2))
    img2 = rgb2.filter(ImageFilter.GaussianBlur(radius = blur_val)) 
    converter = PIL.ImageEnhance.Color(img2)
    img2 = converter.enhance(enh_val)
    converter = PIL.ImageEnhance.Brightness(img2)
    img2 = converter.enhance(bright_val)
    converter = PIL.ImageEnhance.Contrast(img2)
    img2 = converter.enhance(contra_val)
    
    graph1 = make_figure(img2)
    return [graph1, graph2], [{'width': '59%', 'display': 'inline-block', 'padding': '0 20'}, 
                              {'display': 'inline-block', 'width': '39%', 'vertical-align': 'top',
                              'margin-top': '5rem'}], rgb2, clrwhl, cu_sym

                                
                                
@app.callback(
    Output('download', 'children'),
    Input({'type': 'save-data', 'index': ALL}, 'n_clicks'),
    State({'type': 'graph', 'index': ALL}, 'figure'),
    State('image-store', 'data'),
    State({'type': 'image-slider', 'index': ALL}, 'value'),
    prevent_initial_call=True,
)
def func(n_clicks, image, image_store_data, slider_value):
    if any(n_clicks)>0:
        filename = list(image_store_data.keys())[slider_value[0]]
        if len(filename.split('.')) == 2:
            file_type = filename.split('.')[1]
            filename = 'image.' + file_type
        else:
            filename = 'image.tiff'
        
        im_cache = image[0]['data'][0]['source']
        url = "/download/" + urlquote(filename)
        return html.A(download=filename, href=im_cache, children=["Click here to start download"])
    pass
                                
# Run app and display result inline in the notebook
# app.run_server(mode='inline')
app.run_server(mode='external', host='0.0.0.0', debug=True)


# In[ ]:





# In[ ]:




