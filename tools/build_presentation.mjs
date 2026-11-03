/** Editable scientific slides. Run with the bundled Artifact Tool runtime. */
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { FileBlob, Presentation, PresentationFile } from '@oai/artifact-tool';

const ROOT=process.cwd();
const SKILL=process.env.PRESENTATION_SKILL_DIR;
if(!SKILL || !process.env.RUNTIME_PYTHON || !process.env.RUNTIME_NODE_MODULES) throw new Error('Set PRESENTATION_SKILL_DIR, RUNTIME_PYTHON and RUNTIME_NODE_MODULES from the bundled workspace runtime.');
const {applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const B=path.join(ROOT,'results/sequences');
const read=async p=>JSON.parse(await fs.readFile(path.join(ROOT,p),'utf8'));
const data=await read('results/sequences/presentation_data.json');
const s=data.summary, curves=data.curves;
const check=await read('results/sequences/verification.json');
if(check.status!=='passed') throw new Error('Scientific verification required');
const examples=await read('results/sequences/synthetic_examples.json');
const qwen=await read('results/qwen_improved/presentation_examples.json');
const P=Presentation.create({slideSize:{width:1280,height:720}});
const C=['#167D92','#DF8B2F','#7957A0','#CE5351','#4F9A65'];
const LIGHT={'#167D92':'#EAF3F5','#DF8B2F':'#FCF3E9','#7957A0':'#F2EFF6','#CE5351':'#FBEFEE','#4F9A65':'#EEF5F0'};
const INK='#1D3541', MUTED='#526A77', PAPER='#FBFCFD', FONT='Arial';
const row=k=>s.groups.find(r=>r.key===k);
const pct=x=>(100*x).toFixed(2).replace('.',',')+'%';
const num=x=>x.toLocaleString('pt-BR');
const modelNames={bert:'BERT-base',distilbert:'DistilBERT',modernbert:'ModernBERT-base'};
const optNames={sgd:'SGD',momentum:'SGD + momentum',adam:'Adam'};
let tableOwners=[],chartOwners=[],slideNotes=[];

function text(sl,t,x,y,w,h,size=27,color=INK,bold=false,align='left'){
 const sh=sl.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 sh.text=t;sh.text.style={typeface:FONT,fontSize:size,color,bold,alignment:align,verticalAlignment:'top',insets:0,autoFit:'none',wrap:'square'};
 return sh;
}
function add(title,notes='',appendix=false){
 const sl=P.slides.add();sl.background.fill=PAPER;
 text(sl,title,64,42,1152,85,43,INK,true);
 text(sl,`${appendix?'Apoio · ':''}MAE5920`,64,678,900,22,17,MUTED);
 text(sl,String(P.slides.items.length).padStart(2,'0'),1160,678,56,22,17,MUTED,false,'right');
 sl.speakerNotes.text=notes;slideNotes.push({slide:P.slides.items.length,title,notes});
 return sl;
}
function caption(sl,t,y=624){text(sl,t,64,y,1152,44,23,MUTED);}
function block(sl,t,x,y,w,h,color=C[0],size=26){
 const sh=sl.shapes.add({geometry:'rect',name:t.replaceAll('\n',' '),position:{left:x,top:y,width:w,height:h},fill:LIGHT[color]??'#F0F4F6',line:{fill:color,width:1.6}});
 sh.text=t;sh.text.style={typeface:FONT,fontSize:size,color:INK,alignment:'center',verticalAlignment:'middle',insets:9,wrap:'square',autoFit:'none'};
 return sh;
}
function connect(sl,a,b,color=MUTED,from='right',to='left'){
 return sl.shapes.connect(a,b,{kind:'elbow',fromSide:from,toSide:to,line:{fill:color,width:2},tail:{type:'triangle',width:'med',length:'med'}});
}
function table(sl,headers,rows,{x=64,y=175,w=1152,h=385,widths=null,size=25}={}){
 const t=sl.tables.add({rows:rows.length+1,columns:headers.length,left:x,top:y,width:w,height:h,values:[headers,...rows],...(widths?{columnWidths:widths}:{})});
 t.styleOptions={headerRow:true,bandedRows:false};
 t.borders.assign({style:'solid',fill:'#DCE5E9',width:.6});
 for(let r=0;r<=rows.length;r++)for(let c=0;c<headers.length;c++){
  const cell=t.getCell(r,c);cell.fill=r===0?'#E7F1F4':PAPER;
  cell.text.style={typeface:FONT,fontSize:r===0?size-1:size,color:r===0?C[0]:INK,bold:r===0,verticalAlignment:'middle',insets:10,wrap:'square',autoFit:'none'};
 }
 tableOwners.push(P.slides.items.length);return t;
}
function chart(sl,type,{x=64,y=180,w=1152,h=390,categories,series,percent=false,title='',legend=true,ymin=null,ymax=null,scatter=false}={}){
 // Excel workbooks preserve 15 significant digits. Keep a conservative 12 in
 // editable charts; exact scientific values remain in the source JSON/NPZ.
 series=series.map(s=>({...s,...Object.fromEntries(['values','xValues'].filter(k=>s[k]).map(k=>[k,s[k].map(v=>v===null?null:Number(Number(v).toPrecision(12)))]))}));
 const ch=sl.charts.add(type,{position:{left:x,top:y,width:w,height:h},title,categories,series,
  hasLegend:legend,legend:{position:'bottom',overlay:false,textStyle:{fontSize:21,fill:INK}},
  titleTextStyle:{fontSize:25,fill:INK,bold:true},
  chartFill:PAPER,chartLine:{fill:'none',width:0},plotAreaFill:PAPER,plotAreaLine:{fill:'none',width:0},
  xAxis:{textStyle:{fontSize:20,fill:MUTED},line:{fill:'#B5C8D0',width:1},majorGridlines:null},
  yAxis:{numberFormatCode:percent?'0%':'0.00',textStyle:{fontSize:20,fill:MUTED},line:{fill:'none',width:0},majorGridlines:{fill:'#E3EAEE',width:.7},...(ymin!==null?{min:ymin}:{}),...(ymax!==null?{max:ymax}:{})},
  ...(type==='bar'?{barOptions:{direction:'column',grouping:'clustered',gapWidth:130},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:percent?'0.0%':'0.0',textStyle:{fontSize:20,fill:INK}}}:{}),
  ...(type==='scatter'?{scatterOptions:{style:scatter?'marker':'lineWithMarkers'}}:{lineOptions:{smooth:false}})
 });
 applyPresentationChartFont(ch,{fontFamily:FONT});chartOwners.push(P.slides.items.length);return ch;
}
function lineSeries(name,values,color){return {name,values,line:{fill:color,width:2.7},marker:{symbol:'none'}};}

// 1. Minimal typographic cover.
{
 const sl=P.slides.add();sl.background.fill='#153C4B';
 text(sl,'Sentimento, redes\ne treinamento',76,145,1110,210,74,'#FFFFFF',true);
 text(sl,'CNNs, encoders congelados e ajuste completo',80,390,1100,90,35,'#B7DFE7');
 text(sl,'Nestor · Letícia · Gabriel\nMAE5920 · Introdução a Deep Learning',80,570,1080,75,26,'#E6F0F4');
 sl.speakerNotes.text='Objetivo didático: relacionar os classificadores com convoluções, atenção, otimização e generalização. A parte principal acompanha o estudo de texto sequencial. Há material de apoio para MLPs, double descent e steering, cada um com seu próprio protocolo.';
 slideNotes.push({slide:1,title:'Sentimento, redes e treinamento',notes:sl.speakerNotes.text});
}
// 2. Evidence roles.
{
 const sl=add('Amostras e escolhas do experimento','Dados: results/sequences/data_manifest.json e selection_frozen.json. O prefixo de até 128 palavras é comum a todos. CNN e encoders têm limite de 256 tokens e taxas de truncagem próprias; o TF-IDF lê o prefixo completo. A nova auditoria exclui todos os papéis da extensão MLP. O teste oficial não foi reaberto. Selecionamos época e taxa somente pela CE de validação.');
 const a=block(sl,'Treino\n6.144 resenhas',85,210,310,155),b=block(sl,'Validação\n1.536 resenhas',485,210,310,155,C[1]),c=block(sl,'Nova auditoria interna\n2.048 resenhas',885,210,310,155,C[2]);
 text(sl,'Ajuste dos pesos\nVocabulário da CNN e TF-IDF',85,395,310,110,29);
 text(sl,'Escolha de taxa e época\nTrês sementes por condição',485,395,310,110,29);
 text(sl,'Avaliação após congelar\nSem novo ajuste dos modelos',885,395,310,110,29);
 caption(sl,'Mesmo prefixo de 128 palavras. Tokenização e pré-treinamento variam entre famílias.');
}
// 3. Native architecture diagram, dimensions are actual scientific evidence.
{
 const sl=add('Arquitetura da TextCNN','Fonte: src/imdb_lab/models.py e sequence_config.json. Embedding treinável 128, Conv1D paralelas de janelas 3, 4 e 5, cem filtros por janela, ReLU, máximo global com máscara de padding, concatenação de 300, dropout e Linear(300,2). Inspiração: Kim (2014), https://aclanthology.org/D14-1181/. Prince capítulo 10 e slides Redes Convolucionais, pp. 11–22 e 36.');
 const input=block(sl,'Índices\nB × L',64,295,155,115),emb=block(sl,'Embedding\nB × L × 128',270,295,200,115);
 connect(sl,input,emb);
 const out=block(sl,'Concatenar 300\nDropout\nLinear 300 × 2',999,282,220,145,C[3]);
 [3,4,5].forEach((k,i)=>{
  const y=160+i*158;
  const conv=block(sl,`Conv1D ${k} + ReLU\n100 filtros`,530,y,210,100,C[i],25);
  const pool=block(sl,'Máximo global\n100 números',785,y,170,100,C[i],24);
  connect(sl,emb,conv);connect(sl,conv,pool);connect(sl,pool,out);
 });
 caption(sl,`${num(row('cnn_adam').parameters)} parâmetros treináveis. L ≤ 256 tokens. Todo o modelo aprende do zero.`);
}
// 4. Convolution mechanism with an explicitly requested editable diagram.
{
 const sl=add('Convolução: padrões locais com pesos compartilhados','A frase é um exemplo ilustrativo, não uma medição de ativação. O kernel combina embeddings, não regras fixas para palavras. Cada filtro aprende seus pesos e desliza por toda a entrada. ReLU e máximo global resumem a resposta. Uma janela de 3 tokens não equivale a atenção global. Prince capítulo 10 e slides Redes Convolucionais, pp. 12–20.');
 ['not','a','good','movie','at','all'].forEach((t,i)=>block(sl,t,100+i*180,190,150,80,i<3?C[0]:'#8A9DA6',30));
 const kernel=block(sl,'Um filtro de janela 3\nmesmos pesos em cada posição',265,355,720,105,C[0],31);
 text(sl,'Combina os 128 canais de três tokens vizinhos',145,510,990,55,32,INK,true,'center');
 caption(sl,'O máximo global retém a maior resposta do filtro e perde sua posição exata.');
}
// 5. Controlled optimization.
{
 const sl=add('CNN: otimizador e taxa de aprendizado','Dados: results/sequences/runs e presentation_data.json. Linhas são médias de três sementes. Pesos iniciais e minibatches pareados. Doze épocas, batch 64, clipping 1, mesmo dropout. Esquerda: taxa comum 0,001. Direita: cada método usa a melhor média de CE de validação entre três taxas candidatas. Não selecionamos a melhor semente. Faixas completas estão no relatório. Prince 6.1–6.5.');
 chart(sl,'line',{x:64,y:170,w:550,h:420,title:'Taxa comum: 0,001',categories:curves.cnn_common_adam.epochs.map(String),series:Object.keys(optNames).map((o,i)=>lineSeries(optNames[o],curves['cnn_common_'+o].mean,C[i]))});
 chart(sl,'line',{x:662,y:170,w:550,h:420,title:'Taxa selecionada na validação',categories:curves.cnn_adam.validation_ce.epochs.map(String),series:Object.keys(optNames).map((o,i)=>lineSeries(`${optNames[o]} (${s.choices['cnn_'+o]})`,curves['cnn_'+o].validation_ce.mean,C[i]))});
 caption(sl,'Eixo vertical: CE de validação. A mesma taxa e taxas ajustadas respondem perguntas diferentes.');
}
// 6. Generalization and dropout.
{
 const sl=add('CNN: ajuste do treino e overfitting','Dados: results/sequences/presentation_data.json. Curvas médias das três sementes. Dropout desligado para calcular CE de treino e validação. Mesmo Adam e taxa selecionada para a CNN principal. O treino segue até 12 épocas, mas o checkpoint usado na auditoria vem da menor CE de validação. Prince capítulos 8–9; slides Regularização.');
 const keys=['dropout_0','dropout_0.5','dropout_0.8'];
 ['train_ce','validation_ce'].forEach((m,j)=>chart(sl,'line',{x:64+j*598,y:170,w:550,h:420,title:j===0?'CE de treino (modo avaliação)':'CE de validação',categories:curves[keys[0]][m].epochs.map(String),series:keys.map((k,i)=>lineSeries(row(k).label,curves[k][m].mean,C[i]))}));
 caption(sl,'Checkpoint escolhido pela validação. Treinar até a última época e avaliar são decisões distintas.');
}
// 7. Model capacity.
{
 const sl=add('CNN: janelas e quantidade de filtros','Dados: results/sequences/summary.json. Três sementes em cada variante; auditoria interna de 2.048 entradas. A grade de arquiteturas reutiliza a taxa de Adam selecionada para a CNN principal, sem retuning por variante. O embedding contribui com muitos parâmetros, portanto reduzir filtros não reduz toda a rede na mesma proporção.');
 const rs=s.groups.filter(r=>r.group==='cnn_architecture');
 table(sl,['Janelas / filtros','Parâmetros','Acurácia média','Mínimo a máximo'],rs.map(r=>[r.label,num(r.parameters),pct(r.accuracy_mean),`${pct(r.accuracy_min)} a ${pct(r.accuracy_max)}`]),{widths:[410,230,230,282],h:320,size:26});
 text(sl,'Mesmo prefixo, dimensão de embedding e orçamento',70,540,1110,60,32,INK,true);
 caption(sl,'A grade é pequena. O resultado não prova que uma janela seja a melhor em outras tarefas.');
}
// 8. Native encoder diagram.
{
 const sl=add('Representações contextuais no Transformer','Fonte: Prince capítulos 11–12; slides Transformers pp. 7–18 e 21–31; Devlin et al., https://aclanthology.org/N19-1423/. Diagrama funcional: operações e normalização têm ordens distintas entre famílias. Q, K e V têm projeções aprendidas; softmax de compatibilidades controla a combinação de valores. A média usada neste projeto exclui padding e tokens especiais.');
 const a=block(sl,'Embedding\ndos tokens\nB × L × 768',65,282,205,140);
 const b=block(sl,'Bloco Transformer\nAutoatenção\nMLP por posição\nResíduos e normalização',350,205,385,295,C[2],29);
 const c=block(sl,'Média com máscara\nB × 768',820,282,195,140);
 const d=block(sl,'Dropout 0,1\nLinear 768 × 2',1050,282,185,140,C[1],24);
 connect(sl,a,b);connect(sl,b,c);connect(sl,c,d);
 text(sl,'12 / 6 / 22 blocos',375,520,355,55,31,C[2],true,'center');
 caption(sl,'Atenção mistura informação entre tokens. A MLP transforma cada posição.');
}
// 9. Actual model definitions.
{
 const sl=add('Os três encoders treinados','Fontes: models_pinned.json e arquivos *_architecture.json; Devlin et al. (2019), Sanh et al. (2019), https://arxiv.org/abs/1910.01108; Warner et al. (2024), https://arxiv.org/abs/2412.13663; https://huggingface.co/answerdotai/ModernBERT-base. Contagens incluem nossa cabeça de média mais Linear. BERT não inclui pooler. DistilBERT incorpora destilação no pré-treinamento público; não executamos a destilação neste trabalho. ModernBERT usa posições rotativas e atenção local intercalada com global.');
 table(sl,['Encoder','Blocos','Parâmetros','Diferença principal'],[
  ['BERT-base','12',num(s.encoder_architectures.bert.parameters),'Referência bidirecional'],
  ['DistilBERT','6',num(s.encoder_architectures.distilbert.parameters),'Encoder destilado menor'],
  ['ModernBERT-base','22',num(s.encoder_architectures.modernbert.parameters),'RoPE, atenção local e global']
 ],{widths:[280,125,270,477],h:300,size:27});
 text(sl,'768 dimensões e 12 cabeças de atenção',70,530,1110,60,33,INK,true);
 caption(sl,'Checkpoints de pré-treinamento. Pré-treinamento e tokenização diferem entre as famílias.');
}
// 10. Training policies, native diagram.
{
 const sl=add('Encoder congelado e ajuste completo','Dados e política: sequence_config.json, docs/PROTOCOLO_SEQUENCIAS.md. Congelado: encoder em avaliação, vetores pré-computados, cabeça com dropout 0,1, Adam por 40 épocas. Completo: dropout do encoder ativo durante treino, AdamW por 3 épocas, warmup 10%, redução linear e decaimento 0,01. Cabeças compartilham partida por semente. Comparação entre receitas práticas; múltiplos fatores mudam junto da atualização do encoder.');
 for(let i=0;i<2;i++){
  const y=200+i*200;
  text(sl,i===0?'Congelado':'Completo',70,y+20,230,60,32,INK,true);
  const a=block(sl,i===0?'Encoder fixo\nModo avaliação':'Encoder treinável\nRecebe gradientes',320,y,350,125,i===0?C[0]:C[3],29);
  const b=block(sl,'Média e dropout\n768 números',745,y,220,125,C[1],26);
  const c=block(sl,'Cabeça linear\n1.538 pesos e vieses',1040,y,185,125,C[3],24);
  connect(sl,a,b);connect(sl,b,c);
 }
 caption(sl,'Congelado: só a cabeça aprende. Completo: a representação também muda.');
}
// 11. Primary model comparison.
{
 const sl=add('Encoders na nova auditoria interna','Dados: results/sequences/summary.json. Mesmos 2.048 exemplos e prefixos comuns. Cada barra é a média de três sementes após escolher taxa e época na validação. Taxas diferem por encoder e regime. A amostra é interna. Os valores do teste oficial dos relatórios anteriores não entram nesta figura. Probabilidades não calibradas.');
 chart(sl,'bar',{y:175,h:420,title:'Acurácia média das três sementes',categories:['BERT-base','DistilBERT','ModernBERT-base'],series:[
  {name:'Congelado',values:['bert','distilbert','modernbert'].map(k=>row(k+'_probe').accuracy_mean),fill:C[0]},
  {name:'Completo',values:['bert','distilbert','modernbert'].map(k=>row(k+'_full').accuracy_mean),fill:C[1]}
 ],percent:true,ymin:0,ymax:1});
 caption(sl,`Controles: CNN + Adam ${pct(row('cnn_adam').accuracy_mean)}. TF-IDF + logística ${pct(s.baseline.accuracy)}.`);
}
// 12. Main cost evidence.
{
 const sl=add('Qualidade e custo do ajuste','Dados: summary.json, training_seconds_mean, feature_extraction_seconds e audit_end_to_end_seconds. A coluna de ajuste exclui avaliações entre épocas. Congelado adiciona extração dos vetores de treino e validação. A inferência inclui o encoder também no modo congelado, batch 64. Download, tokenização e carregamento excluídos. NVIDIA GB10, BF16 para encoders, FP32 para CNN. Medição descritiva desta implementação, sem otimizar kernels por família.');
 const rs=['distilbert_probe','distilbert_full','bert_probe','bert_full','modernbert_probe','modernbert_full'].map(row);
 table(sl,['Modelo / regime','Acurácia','Ajuste + vetores','Inferência / 2.048'],rs.map(r=>[r.label,pct(r.accuracy_mean),r.fit_plus_features_seconds.toFixed(1)+' s',r.audit_end_to_end_seconds.toFixed(1)+' s']),{widths:[465,200,245,242],h:415,size:25});
 caption(sl,'Congelar economiza gradientes, mas continua exigindo o encoder na inferência.');
}
// 13. Synthetic illustration with common source text.
{
 const sl=add('Um mesmo texto, decisões diferentes','Texto próprio: synthetic_examples.json, synthetic_reversal. Os modelos usam a semente 17, sem escolher pela auditoria. Texto definido antes de observar as predições e avaliado após o congelamento dos modelos. Intenção positiva do autor, sem constituir benchmark. As probabilidades não foram calibradas.');
 const e=examples.examples.find(x=>x.id==='synthetic_reversal');
 text(sl,e.text,75,155,1130,195,31,INK);
 table(sl,['CNN','BERT completo','DistilBERT completo','ModernBERT completo'],[['cnn','bert_full','distilbert_full','modernbert_full'].map(k=>pct(e.probabilities[k]))],{y:405,h:180,size:28});
 caption(sl,'Valores: P(positivo). A negação inicial exige considerar a reversão no restante do texto.');
}
// 14. Course connections.
{
 const sl=add('O que relacionamos com as aulas','Fontes: docs/REFERENCIAS_SEQUENCIAS.json, docs/REFERENCIAS_DISCIPLINA.json, resultados originais e extensões. O double descent é uma regressão sintética com camada aleatória congelada. O steering altera ativações de Qwen3-8B com pesos congelados. Esses mecanismos não foram testados como melhorias da CNN.');
 table(sl,['Conceito','Evidência do trabalho'],[
  ['Convolução e pooling','Janelas locais, pesos compartilhados e máximos dos filtros'],
  ['Otimização','Mesma taxa e seleção de taxa mudam a comparação'],
  ['Generalização','Curvas e checkpoint por validação, sem escolher semente'],
  ['Transferência','Cabeça linear sobre vetores fixos e ajuste do encoder'],
  ['Representações internas','Steering do Qwen e demonstrações sintéticas próprias']
 ],{widths:[345,807],h:405,size:27});
 caption(sl,'A contribuição didática é explicar o mecanismo e os limites de cada experimento.');
}
// 15. Full hyperparameter grid, appendix.
{
 const sl=add('Grades de hiperparâmetros','Fonte: sequence_config.json. Três sementes 17, 29, 43 em toda condição neural. Seleção da taxa pela média da menor CE de validação por execução; checkpoint escolhe a primeira época com mínimo. Sementes publicadas integralmente.',true);
 table(sl,['Experimento','Taxas','Épocas / batch'],[
  ['CNN · SGD','0,001 / 0,03 / 0,3','12 / 64'],
  ['CNN · momentum 0,9','0,001 / 0,01 / 0,1','12 / 64'],
  ['CNN · Adam','0,0003 / 0,001 / 0,003','12 / 64'],
  ['Encoders · cabeça','0,001 / 0,01 / 0,1','40 / 128'],
  ['Encoders · completo','0,00001 / 0,00003','3 / 32']
 ],{widths:[380,480,292],h:395,size:27});
 caption(sl,'87 treinos neurais. Janelas e dropout da CNN reutilizam a taxa selecionada para Adam.');
}
// 16. Fine-tuning trajectories.
{
 const sl=add('Curvas do ajuste completo','Fonte: results/sequences/presentation_data.json. Curvas médias de três sementes com a taxa escolhida separadamente para cada encoder. Época zero usa cabeça aleatória e não participa da seleção. CE de validação, não acurácia de auditoria, escolhe taxa e época.',true);
 chart(sl,'line',{title:'CE de validação',categories:curves.bert_full.validation_ce.epochs.map(String),series:['bert','distilbert','modernbert'].map((k,i)=>lineSeries(modelNames[k],curves[k+'_full'].validation_ce.mean,C[i])),ymin:0,ymax:.85});
 caption(sl,'Três épocas são um orçamento fixo. O experimento não garante convergência.');
}
// 17. Seeds, actual metrics.
{
 const sl=add('Variação entre as três sementes','Fonte: summary.json. A faixa é mínimo a máximo entre três inicializações na mesma partição. Não é intervalo de confiança amostral. Todas as execuções e configurações descartadas estão publicadas. Auditoria interna de 2.048 exemplos.',true);
 table(sl,['Modelo / regime','Média','Mínimo','Máximo'],s.groups.filter(r=>r.group==='encoder').map(r=>[r.label,pct(r.accuracy_mean),pct(r.accuracy_min),pct(r.accuracy_max)]),{widths:[540,204,204,204],h:415,size:26});
 caption(sl,'Uma única partição e três sementes descrevem somente parte da incerteza.');
}
// 18. Exact CNN output readout, editable table.
{
 const sl=add('Como a última camada lê a CNN','Fonte: synthetic_examples.json. Margem logit positivo menos negativo = viés + soma dos máximos dos filtros vezes a diferença dos pesos da saída. A decomposição é exata com dropout desligado. Agrupamos filtros que ganharam na mesma janela. Isso não identifica efeito causal de uma palavra: alterar a entrada pode mudar outros máximos.',true);
 const r=examples.examples.find(x=>x.id==='synthetic_reversal').cnn_readout;
 text(sl,'Margem = viés + soma das contribuições',72,150,1110,65,36,INK,true);
 table(sl,['Janela vencedora','Tamanho','Contribuição à margem'],[...r.positive_windows.slice(0,2),...r.negative_windows.slice(0,2)].map(x=>[x.words,String(x.kernel),x.contribution.toFixed(3)]),{y:245,widths:[600,170,382],h:300,size:28});
 caption(sl,`Quatro janelas ilustradas. Margem total e reconstruída: ${r.logit_margin.toFixed(4)}.`);
}
// 19. Earlier didactic double descent, native evidence chart.
{
 const sl=add('Double descent no exemplo sintético','Estudo separado, anterior aos treinos de CNN/encoders: results/didactic/double_descent.csv. Vinte sementes prescritas, n=64, camada ReLU aleatória congelada. Ajustamos p coeficientes de saída por norma mínima ou ridge lambda 0,001. Cada largura p ocupa uma categoria: o espaçamento horizontal não representa diferenças numéricas entre larguras. O eixo vertical mostra log10 da mediana do MSE, calculada sobre todas as vinte sementes. O pico ocorre em p=64. O relatório próprio mostra o eixo quantitativo p/n, faixas interquartis e todas as 880 condições. Não demonstramos double descent na CNN. Fonte: Prince 8.4 e Belkin et al. (2019), https://arxiv.org/abs/1812.11118.',true);
 const raw=(await fs.readFile(path.join(ROOT,'results/didactic/double_descent.csv'),'utf8')).trim().split('\n');
 const header=raw.shift().split(',');const rows=raw.map(l=>Object.fromEntries(l.split(',').map((v,i)=>[header[i],Number(v)])));
 const median=a=>{a.sort((a,b)=>a-b);const m=a.length/2;return (a[m-1]+a[m])/2;};
 const widths=[...new Set(rows.map(r=>r.p))].sort((a,b)=>a-b);
 chart(sl,'line',{x:64,y:165,w:1152,h:420,title:'Log10 da mediana do MSE',categories:widths.map(String),series:[0,.001].map((lambda,i)=>({name:lambda===0?'Sem ridge':'Ridge 0,001',values:widths.map(p=>Math.log10(median(rows.filter(r=>r.p===p&&r.lambda===lambda).map(r=>r.test_mse)))),line:{fill:C[i],width:2.5},marker:{symbol:'circle',size:5,fill:C[i]}}))});
 caption(sl,'Larguras p testadas (eixo categórico). n = 64. Vertical: log10 do MSE mediano.');
}
// 20. Actual existing steering example, no new Qwen inference.
{
 const sl=add('Steering no Qwen3-8B congelado','Estudo anterior, sem nova inferência do Qwen: results/qwen_improved/presentation_examples.json, primeira fonte sintética do conjunto reservado, sem seleção por resultado. Direção unitária pela diferença de médias dos estados de 128 fontes com instruções positivas e negativas no bloco 28. h novo = h + alpha vezes escala vezes direção. Pesos e prompt permanecem iguais. A intervenção começa depois do marcador REVIEW. O prefixo FACT tem tokens idênticos nas três condições. O objetivo é demonstrar uma intervenção nas ativações, sem provar superioridade sobre prompting.',true);
 const q=qwen.cases[0];
 text(sl,'h′ = h + α · escala · direção',75,145,1120,70,43,INK,true);
 text(sl,'Enredo comum: Bela e Evan organizam uma apresentação no teatro de uma vila.',75,225,1120,55,26,INK);
 const qs=[-2,0,2].map(g=>{const r=q.outputs.find(o=>o.gain===g);return [g===0?'Sem intervenção':g<0?'Intervenção negativa':'Intervenção positiva',r.generated.split('REVIEW:').at(-1).trim()];});
 table(sl,['Condição','REVIEW gerada'],qs,{y:290,h:320,widths:[330,822],size:24});
 caption(sl,'Mesmo prompt e pesos. Intervenção nas ativações depois de REVIEW. Prefixo factual idêntico.');
}
// 21. Sources and scope.
{
 const sl=add('Fontes e limites','Referências completas: docs/REFERENCIAS_SEQUENCIAS.json e docs/REFERENCIAS_DISCIPLINA.json. Fontes externas primárias: Kim https://aclanthology.org/D14-1181/; Devlin https://aclanthology.org/N19-1423/; Sanh https://arxiv.org/abs/1910.01108; Warner https://arxiv.org/abs/2412.13663; modelo https://huggingface.co/answerdotai/ModernBERT-base. Diagramas e gráficos próprios. Nenhuma página dos PDFs de referência foi redistribuída.',true);
 text(sl,'Prince, Understanding Deep Learning\nCapítulos 6 a 12',72,165,1120,95,31,INK,true);
 text(sl,'Slides do acervo do grupo\nOtimização, generalização, regularização, CNNs e Transformers',72,295,1120,100,29);
 text(sl,'Kim (2014), Devlin et al. (2019), Sanh et al. (2019)\nWarner et al. (2024) e documentação oficial',72,430,1120,95,27);
 caption(sl,'Resultados internos, grades pequenas e uma partição. Arquitetura e pré-treinamento variam juntos.');
}

// Preserve all charts as native editable evidence with literal-data workbook snapshots.
const TMP=path.join(ROOT,'.cache/presentation_build');
await fs.mkdir(TMP,{recursive:true});await fs.mkdir(path.join(TMP,'validated'),{recursive:true});await fs.mkdir(path.join(ROOT,'output/slides'),{recursive:true});
let rev=1;while(true){try{await fs.access(path.join(TMP,`candidate-${rev}.pptx`));rev++;}catch{break;}}
const candidate=path.join(TMP,`candidate-${rev}.pptx`),validated=path.join(TMP,'validated',`seminario-${rev}.pptx`);
await(await PresentationFile.exportPptx(P)).save(candidate);
const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:validated,
 pythonExecutable:process.env.RUNTIME_PYTHON,
 integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--cover-role','cover','--validate-bullet-geometry','--validate-heading-fit',...[...new Set(tableOwners)].flatMap(n=>['--require-native-table-slide',String(n)])],
 fontPolicy:{basis:'design',families:[FONT]},explicitTotalSlideCount:21,
 requiredNativeTableOwnerSlides:[...new Set(tableOwners)],requiredNativeChartOwnerSlides:[...new Set(chartOwners)],
 materializeLiteralChartWorkbooks:true,verifyArtifactToolImport:true,
 receiptPath:path.join(TMP,`validation-${rev}.json`)
});
const final=path.join(ROOT,'output/slides/seminario_mae5920.pptx');await fs.copyFile(validated,final);
const previews=path.join(TMP,'previews');await fs.mkdir(previews,{recursive:true});
const reviewed=await PresentationFile.importPptx(await FileBlob.load(final));
if(reviewed.slides.items.length!==21)throw new Error('Unexpected final slide count on import');
for(const [i,slide] of reviewed.slides.items.entries()){
 const blob=await reviewed.export({slide,format:'png',scale:1});
 await fs.writeFile(path.join(previews,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const digest=createHash('sha256').update(await fs.readFile(final)).digest('hex');
await fs.writeFile(path.join(B,'presentation_manifest.json'),JSON.stringify({pptx:'output/slides/seminario_mae5920.pptx',sha256:digest,
 slides:21,main_slides:'1-14',appendix_slides:'15-21',native_charts:[...new Set(chartOwners)],native_tables:[...new Set(tableOwners)],
 architecture_diagrams:'native editable shapes and attached connectors',font_family:FONT,chart_precision:'12 significant digits for editable workbook compatibility; unrounded scientific source retained in JSON/NPZ',double_descent_display:'Categorical tested widths p, vertical log10(median MSE); all twenty seeds and widths retained',
 package_validation:'passed',visual_review:'pending',builder_sha256:createHash('sha256').update(await fs.readFile(path.join(ROOT,'tools/build_presentation.mjs'))).digest('hex'),data_sha256:createHash('sha256').update(await fs.readFile(path.join(B,'presentation_data.json'))).digest('hex')},null,2)+'\n');
await fs.writeFile(path.join(ROOT,'docs/NOTAS_APRESENTACAO.md'),'# Notas da apresentação\n\nOs slides 1–14 formam o roteiro principal. Os slides 15–21 são apoio. Para uma fala curta, priorize 1–3, 5–6 e 9–14.\n\n'+slideNotes.map(n=>`## ${n.slide}. ${n.title}\n\n${n.notes}\n`).join('\n'));
console.log(JSON.stringify({slides:P.slides.items.length,file:'output/slides/seminario_mae5920.pptx',sha256:digest,validated:true}));
