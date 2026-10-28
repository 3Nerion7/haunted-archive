"""Render a captioned, explicitly edited replay from recorded CLI JSON. No inference."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess
from PIL import Image,ImageDraw,ImageFont

W,H=1280,720
BG='#10141d';PANEL='#1b2330';INK='#f1ead9';MUTED='#b8c5d3';GOLD='#ebc17b';MINT='#8fd5bd';RED='#efb2ab'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fonts():
 candidates=[Path('C:/Windows/Fonts/segoeui.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]
 p=next((p for p in candidates if p.exists()),None)
 if p is None:raise SystemExit('Provide an installed Segoe UI or DejaVu Sans font; no download performed.')
 return p
FONT=fonts()
def f(n):return ImageFont.truetype(str(FONT),n)
def wrap(draw,text,size,width):
 lines=[]
 for para in text.split('\n'):
  line=''
  for word in para.split():
   candidate=(line+' '+word).strip()
   if line and draw.textlength(candidate,font=f(size))>width:lines.append(line);line=word
   else:line=candidate
  lines.append(line)
 return lines
def text(draw,xy,value,size=26,color=INK,width=1140,max_height=500):
 lines=wrap(draw,value,size,width);height=(size+9)*len(lines)
 assert height<=max_height,(value,height,max_height)
 x,y=xy
 for line in lines:draw.text((x,y),line,font=f(size),fill=color);y+=size+9
 return y
def canvas(index,title):
 im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
 d.rounded_rectangle((40,30,1240,76),radius=12,fill=PANEL)
 text(d,(58,38),'HAUNTED ARCHIVE  /  CAPTIONED CLI REPLAY',22,GOLD,max_height=32)
 text(d,(58,103),title,46,INK,max_height=120)
 d.line((58,647,1222,647),fill='#3c4b60',width=2)
 text(d,(58,663),'Edited replay of actual stdout • Fictional inputs • Screen durations are not process timings',20,MUTED,max_height=32)
 d.rectangle((58,707,58+int(1164*(index+1)/6),711),fill=MINT)
 return im,d
def card(d,x,y,width,height,label,body,color=MINT,size=26):
 d.rounded_rectangle((x,y,x+width,y+height),radius=16,fill=PANEL,outline='#3c4b60',width=2)
 text(d,(x+22,y+18),label,23,color,width=width-44,max_height=65)
 text(d,(x+22,y+80),body,size,INK,width=width-44,max_height=height-96)
def main():
 p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=Path,required=True);p.add_argument('--ffmpeg',default=shutil.which('ffmpeg'));a=p.parse_args()
 assert a.ffmpeg,'An already installed ffmpeg is required'
 a.output.mkdir(parents=True,exist_ok=True);a.frames.mkdir(parents=True,exist_ok=True)
 assert not (a.output/'demo.mp4').exists(),'Inspect existing render before replacing it'
 r=json.loads(a.receipt.read_text(encoding='utf-8'));by={x['id']:x for x in r['records']}
 assert len(by)==5 and all(x['exit']==0 for x in by.values())
 for rec in by.values():assert hashlib.sha256(rec['stdout_utf8'].encode('utf-8')).hexdigest()==rec['stdout_sha256']
 sem=by['paraphrase-semantic']['result'];lex=by['paraphrase-lexical']['result'];lan=by['lantern']['result'];missing=by['missing-wattage']['result'];note=by['notebook']['result']
 assert sem['hits'][0]['source_id']=='voices' and lex['abstained'] and not missing['abstained']
 lantern={h['source_id']:h for h in lan['hits']};assert {'harbor-a','harbor-b','glasshouse'}<=set(lantern)
 assert note['supports_hypothesis']=='NOT_AUTOMATICALLY_DETERMINED' and note['personal_review']=='NOT_CLAIMED'
 scenes=[];captions=[]
 im,d=canvas(0,'Two witnesses. One lantern.')
 text(d,(65,221),'A local clue notebook built around open-weight MiniLM embeddings.',34,INK,max_height=100)
 text(d,(65,332),'Ask a question. Inspect the passages. Save a hypothesis with two exact sources.',29,MINT,width=1100,max_height=110)
 text(d,(65,491),'Five original fictional records · Windows CPU execution · No story answers generated',25,MUTED,width=1100,max_height=90)
 scenes.append((im,7));captions.append({'scene':0,'seconds':7,'caption':'A limited CLI clue notebook with five fictional passages. The model ranks clues and generates no answer.'})
 im,d=canvas(1,'01  /  Search by meaning')
 text(d,(60,187),sem['query'],29,GOLD,max_height=78)
 top=sem['hits'][0]
 card(d,60,273,565,306,f'MiniLM · cosine {top["score"]:.6f}',top['quote'],size=26)
 card(d,651,273,565,306,'TF-IDF lexical comparison · 0 hits','This single paraphrase finds a semantic clue without a lexical match. The cutoffs differ: 0.40 versus >0. This is not an accuracy study.',size=26)
 text(d,(65,595),'Captured commands: rank QUERY   /   rank QUERY --mode lexical',22,MUTED,max_height=35)
 scenes.append((im,14));captions.append({'scene':1,'seconds':14,'record_ids':['paraphrase-semantic','paraphrase-lexical'],'caption':f'Exact query: {sem["query"]} Semantic voices score {top["score"]:.6f}; lexical retrieval abstains. One illustration, not superiority.'})
 im,d=canvas(2,'02  /  Inspect incompatible accounts')
 text(d,(60,185),lan['query'],28,GOLD,max_height=78)
 for x,id_ in [(60,'harbor-a'),(651,'harbor-b')]:
  hit=lantern[id_];card(d,x,265,565,264,hit['source_id']+' · exact source quote',hit['quote'],color=MINT if id_=='harbor-a' else RED,size=25)
 text(d,(65,551),'Conflict flags compare author-written annotations, not model reasoning.',24,INK,max_height=42)
 text(d,(65,593),f'Known false positive retained: greenhouse also returned, cosine {lantern["glasshouse"]["score"]:.6f}.',23,RED,max_height=38)
 scenes.append((im,16));captions.append({'scene':2,'seconds':16,'record_ids':['lantern'],'caption':f'Both exact lantern passages are displayed. Conflicts come from authored metadata; irrelevant glasshouse hit {lantern["glasshouse"]["score"]:.6f} remains visible.'})
 im,d=canvas(3,'03  /  A missing fact still finds a match')
 text(d,(60,186),missing['query'],29,GOLD,max_height=78)
 scores='\n'.join(h['source_id']+'  '+format(h['score'],'.6f') for h in missing['hits'])
 card(d,60,275,420,260,'Actual retrieval result',scores,size=30)
 card(d,506,275,710,260,'The archive never states a wattage','Related passages were returned instead of abstaining. No wattage answer is generated. Similarity does not show that a question is answerable.',color=RED,size=28)
 text(d,(65,570),'These are this exact recorded query’s scores, not scores from a differently worded question.',23,MUTED,width=1110,max_height=65)
 scenes.append((im,14));captions.append({'scene':3,'seconds':14,'record_ids':['missing-wattage'],'caption':'The exact wattage question still retrieves related passages; displayed scores come from this execution. No answer is generated and no reliable answerability detector is claimed.'})
 im,d=canvas(4,'04  /  Keep the reader’s uncertainty')
 text(d,(60,193),note['hypothesis'],29,GOLD,width=1100,max_height=90)
 card(d,60,300,565,266,'Two selected sources','harbor-a + harbor-b\nLiteral quotations, source IDs, offsets and SHA256 hashes are preserved.',size=28)
 card(d,651,300,565,266,'Actual notebook fields','supports_hypothesis:\nNOT_AUTOMATICALLY_DETERMINED\npersonal_review: NOT_CLAIMED',size=25)
 text(d,(65,593),'Fictional test input saved to a new JSON file; no human playtest is represented.',23,MUTED,max_height=36)
 scenes.append((im,18));captions.append({'scene':4,'seconds':18,'record_ids':['notebook'],'caption':'The supplied fictional hypothesis is saved with two exact sources. The actual output does not automatically determine support and claims no personal review.'})
 im,d=canvas(5,'Reproduce the interaction')
 text(d,(65,220),'Python CLI + local MiniLM + ONNX Runtime',34,MINT,max_height=55)
 text(d,(65,309),'Source, commands, complete JSON and model hashes are available in the repository. Weights and dependencies are not bundled.',28,INK,width=1100,max_height=135)
 text(d,(65,474),'github.com/3Nerion7/haunted-archive',32,GOLD,max_height=60)
 text(d,(65,552),'A small fictional prototype. No accuracy, novelty, prize or revenue guarantee.',24,MUTED,max_height=55)
 scenes.append((im,7));captions.append({'scene':5,'seconds':7,'caption':'Reproduction instructions and full records accompany the source. Runtime/model prerequisites are separate. This is a limited fictional prototype, not a complete game or accuracy claim.'})
 frames=[]
 for i,(im,seconds) in enumerate(scenes):
  path=a.frames/f'{i:02}.png';im.save(path);frames.append((path,seconds))
 scenes[0][0].save(a.output/'demo-poster.png')
 concat=a.frames/'frames.txt';concat.write_text(''.join("file '"+str(p.resolve()).replace('\\','/')+"'\nduration "+str(sec)+'\n' for p,sec in frames)+"file '"+str(frames[-1][0].resolve()).replace('\\','/')+"'\n",encoding='utf-8',newline='\n')
 duration=sum(sec for _,sec in frames)
 cmd=[a.ffmpeg,'-hide_banner','-loglevel','error','-f','concat','-safe','0','-i',str(concat),'-vf','fps=24','-t',str(duration),'-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart','-metadata','title=Haunted Archive - captured CLI replay','-metadata','comment=Edited replay of real captured stdout. Fictional data. Screen duration is not process latency.',str(a.output/'demo.mp4')]
 done=subprocess.run(cmd,capture_output=True,timeout=180);assert done.returncode==0,done.stderr.decode('utf-8',errors='replace')
 script={'video':'demo.mp4','format':'1280x720 H264,24fps,silent captioned edited CLI replay','editorial_duration_seconds':duration,'not_process_latency':True,'receipt_sha256':digest(a.receipt),'scenes':captions,'input_is_actual_recorded_stdout':True,'no_live_interface_claim':True,'all_passages_and_hypothesis_are_fictional':True}
 (a.output/'demo-script.txt').write_bytes((json.dumps(script,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
 print(json.dumps({'duration_seconds':duration,'video_bytes':(a.output/'demo.mp4').stat().st_size,'video_sha256':digest(a.output/'demo.mp4'),'poster_sha256':digest(a.output/'demo-poster.png'),'script_sha256':digest(a.output/'demo-script.txt')}))
if __name__=='__main__':main()
