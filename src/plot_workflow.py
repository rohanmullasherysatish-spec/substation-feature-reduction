"""Render the portrait study workflow as a PNG for the methodology chapter."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "figures/methodology").mkdir(parents=True, exist_ok=True)

FONT = '/System/Library/Fonts/Supplemental/Arial.ttf'
BOLD = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'

S = 3
im = Image.new('RGB', (1200*S, 1600*S), 'white')
d = ImageDraw.Draw(im)
def font(n, bold=False): return ImageFont.truetype(BOLD if bold else FONT, round(n*S))
def rect(box, fill, outline='#455A64', radius=14, width=2):
    d.rounded_rectangle(tuple(round(v*S) for v in box), radius=radius*S, fill=fill, outline=outline, width=width*S)
def text(x,y,t,size=28,bold=False):
    d.multiline_text((x*S,y*S),t,font=font(size,bold),fill='#1E1E1E',anchor='mm',align='center',spacing=6*S)
def box(x,y,w,h,t,fill='white',size=28):
    rect((x,y,x+w,y+h),fill)
    text(x+w/2,y+h/2,t,size)
def arrow(points):
    ps=[(round(x*S),round(y*S)) for x,y in points]
    d.line(ps, fill='#455A64',width=2*S,joint='curve')
    (x0,y0),(x1,y1)=points[-2:]
    vx,vy=x1-x0,y1-y0
    length=(vx*vx+vy*vy)**.5
    ux,uy=vx/length,vy/length
    a=(x1-10*ux+5*uy,y1-10*uy-5*ux)
    b=(x1-10*ux-5*uy,y1-10*uy+5*ux)
    d.line([(round(a[0]*S),round(a[1]*S)),ps[-1],(round(b[0]*S),round(b[1]*S))],fill='#455A64',width=2*S)

rect((25,500,575,1245),'#EAF1F6',radius=16)
rect((625,500,1175,1245),'#F4F6F8',radius=16)
text(600,28,'Applied separately to IEC 104 and IEC 61850',30,True)
text(300,532,'Grouped development validation',28,True)
text(900,532,'All-development fitting',28,True)
box(320,65,560,100,'Prepare protocol-specific data')
box(320,210,560,100,'Group and allocate observations')
box(50,365,480,100,'Development observations')
box(670,365,480,100,'Reserved test observations','#FFF5E4')
text(195,324,'80% target',26)
text(1010,324,'20% target',26)
box(50,580,480,105,'Fit training preprocessing\nand full-model candidates')
box(50,735,480,120,'Score three validation folds\nand select full-model settings')
box(50,910,480,130,'Rank training fields\nSHAP / MI / matched MI\nRefit at 25% and 50%')
box(50,1095,480,105,'Evaluate held-fold predictions\nCompare fold ranks')
box(670,580,480,105,'Fit all-development\npreprocessing')
box(670,735,480,120,'Fit selected full models\nand dummy reference')
box(670,910,480,130,'Rank development fields\nSHAP / MI / matched MI\nRefit at 25% and 50%')
box(670,1095,480,105,'Save 26 primary pipelines\nand 10 matched-MI pipelines')
box(240,1310,720,105,'Compare fixed test predictions\nPerformance, classes and costs')
box(240,1475,720,105,'Exploratory diagnostics\nErrors, encoded matches and 10-round timing\nSelection overhead and cost offsets',size=25)
routes=[[(600,165),(600,210)],[(600,310),(600,340),(290,340),(290,365)],[(600,340),(910,340),(910,365)],[(290,465),(290,480),(15,480),(15,632.5),(50,632.5)],[(530,415),(600,415),(600,632.5),(670,632.5)],[(290,685),(290,735)],[(290,855),(290,910)],[(290,1040),(290,1095)],[(530,795),(670,795)],[(910,685),(910,735)],[(910,855),(910,910)],[(910,1040),(910,1095)],[(910,1200),(910,1270),(600,1270),(600,1310)],[(1150,415),(1190,415),(1190,1362.5),(960,1362.5)],[(600,1415),(600,1475)],[(50,1147.5),(15,1147.5),(15,1527.5),(240,1527.5)]]
for route in routes: arrow(route)
im.save(ROOT/'figures/methodology/study_workflow.png',dpi=(600,600),optimize=True)

