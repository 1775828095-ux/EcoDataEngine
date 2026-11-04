"""Bounded, non-executing data-flow preview of C-shell input declarations.

This is not a shell. Only assignments, predicates, finite loops and date/integer
expressions are interpreted. Commands never run; unsupported values stay UNKNOWN.
"""
from dataclasses import dataclass
from datetime import datetime,timedelta
import ast,copy,re,shlex

@dataclass(frozen=True)
class Value:
    text: str
    unknown: frozenset=frozenset()


def logical_lines(text):
    text=re.sub(r'\\\r?\n',' ',text);result=[];heredoc=None
    for number,line in enumerate(text.splitlines(),1):
        if heredoc:
            if line.strip()==heredoc:heredoc=None
            continue
        if re.search(r'<<\s*[\w\'"]+',line):
            heredoc=re.search(r'<<\s*[\'"]?(\w+)',line)[1];continue
        quote=None;tick=False;parts=[];start=0
        for i,c in enumerate(line):
            if c=='`' and quote!="'":tick=not tick
            if c in '\'"' and not tick:
                if quote==c:quote=None
                elif quote is None:quote=c
            if not quote and not tick and c=='#':line=line[:i];break
        quote=None;tick=False
        for i,c in enumerate(line):
            if c=='`' and quote!="'":tick=not tick
            if c in '\'"' and not tick:
                if quote==c:quote=None
                elif quote is None:quote=c
            if c==';' and not quote and not tick:parts.append(line[start:i]);start=i+1
        parts.append(line[start:])
        result.extend((number,p.strip()) for p in parts if p.strip())
    return result


def condition_head(line):
    start=line.index('(');depth=0;quote=None
    for i in range(start,len(line)):
        c=line[i]
        if c in '\'"':
            if quote==c:quote=None
            elif quote is None:quote=c
        if quote:continue
        if c=='(':depth+=1
        if c==')':
            depth-=1
            if not depth:return line[start+1:i],line[i+1:].strip()
    raise ValueError('未闭合的控制条件：'+line)


def parse(lines):
    def block(i,stops=()):
        nodes=[]
        while i<len(lines):
            n,s=lines[i]
            if s.split()[0] in stops:return nodes,i
            if re.match(r'^if\s*\(',s):
                node,i=branch(i,s);nodes.append(node);continue
            if s.startswith(('while ','while(','foreach ')):
                body,j=block(i+1,('end',))
                if j==len(lines):raise ValueError('控制循环缺end')
                nodes.append(('loop',n,s,body));i=j+1;continue
            nodes.append(('command',n,s));i+=1
        return nodes,i
    def branch(i,s):
        n=lines[i][0];condition,tail=condition_head(s)
        if tail!='then':return ('if',n,condition,[('command',n,tail)],[]),i+1
        yes,j=block(i+1,('else','endif'));no=[]
        if j==len(lines):raise ValueError('控制条件缺endif')
        if lines[j][1].startswith('else if'):
            child,k=branch(j,lines[j][1][5:]);return ('if',n,condition,yes,[child]),k
        if lines[j][1]=='else':no,j=block(j+1,('endif',))
        if j==len(lines) or lines[j][1]!='endif':raise ValueError('控制条件缺endif')
        return ('if',n,condition,yes,no),j+1
    return block(0)[0]


class ShellDeclarations:
    def __init__(self,files,limit=100000,loop_limit=366):
        self.files=files;self.limit=limit;self.loop_limit=loop_limit
        self.steps=0;self.trace=[];self.frames=[];self.includes=[];self.active=[];self.invocation_seen=False

    def expand(self,s,env):
        missing=set()
        def replace(m):
            key=m[1] or m[2];index=m[3]
            v=env.get(key,Value('${'+key+'}',frozenset({key})));missing.update(v.unknown)
            if index:
                try:return shlex.split(v.text)[int(index)-1]
                except (ValueError,IndexError):missing.add(key);return '${'+key+'}'
            return v.text
        # Single-quoted strings do not expand shell variables.
        parts=re.split(r"('[^']*')",s)
        out=''.join(p if p.startswith("'") else re.sub(r'\$\{(\w+)\}|\$(\w+)(?:\[(\d+)\])?',replace,p) for p in parts)
        return Value(out,frozenset(missing))

    def number(self,s):
        tree=ast.parse(s,mode='eval')
        def walk(n):
            if isinstance(n,ast.Constant) and type(n.value) in (int,float):return n.value
            if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.USub,ast.UAdd)):return (-1 if isinstance(n.op,ast.USub) else 1)*walk(n.operand)
            if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.FloorDiv,ast.Mod)):
                a,b=walk(n.left),walk(n.right)
                if isinstance(n.op,ast.Add):return a+b
                if isinstance(n.op,ast.Sub):return a-b
                if isinstance(n.op,ast.Mult):return a*b
                if isinstance(n.op,(ast.Div,ast.FloorDiv)):return a//b if isinstance(n.op,ast.FloorDiv) else a/b
                return a%b
            raise ValueError('不支持的整数表达式')
        v=walk(tree.body)
        if abs(v)>1e12:raise ValueError('整数表达式超出预算')
        return str(int(v)) if int(v)==v else str(v)

    def value(self,s,env):
        expanded=self.expand(s,env)
        if expanded.unknown:return expanded
        if '`' in expanded.text:
            match=re.fullmatch(r'`(.*)`',expanded.text.strip())
            if match:
                command=match[1]
                try:
                    tokens=shlex.split(command)
                    if tokens[:2] in (['date','-ud'],['date','-d']) and len(tokens)==4:
                        date=re.fullmatch(r'(\d{4}-\d{2}-\d{2}|\d{7})([+-]\d+days?)?',tokens[2])
                        if date and re.fullmatch(r'(?:%[YymdjHMS]|[-:])+',tokens[3][1:]):
                            base=datetime.strptime(date[1],'%Y-%m-%d' if '-' in date[1] else '%Y%j')
                            if date[2]:base+=timedelta(days=int(re.match(r'[+-]\d+',date[2])[0]))
                            return Value(base.strftime(tokens[3][1:]))
                    arithmetic=re.fullmatch(r'echo\s+[\'"]?([^|]+?)[\'"]?\s*\|\s*bc(?:\s+-l)?',command)
                    if arithmetic:return Value(self.number(arithmetic[1].strip().strip('\'"')))
                    if len(tokens)==2 and tokens[0]=='seq' and tokens[1].isdigit() and int(tokens[1])<=self.loop_limit:
                        return Value(' '.join(map(str,range(1,int(tokens[1])+1))))
                except (ValueError,ZeroDivisionError):pass
            return Value('${UNSUPPORTED_COMMAND}',frozenset({'command substitution'}))
        try:return Value(' '.join(shlex.split(expanded.text)))
        except ValueError:return Value('${UNSUPPORTED_VALUE}',frozenset({'shell value'}))

    def predicate(self,s,env,control):
        s=s.strip()
        if s.startswith('(') and condition_head('if '+s)[1]=='':return self.predicate(s[1:-1],env,control)
        # Split boolean operators only outside parentheses/quoted literals.
        for op in ('||','&&'):
            depth=0;quote=None
            for i,c in enumerate(s):
                if c in '\'"':
                    if quote==c:quote=None
                    elif quote is None:quote=c
                if quote:continue
                depth+=int(c=='(')-int(c==')')
                if depth==0 and s[i:i+2]==op:
                    a=self.predicate(s[:i],env,control);b=self.predicate(s[i+2:],env,control)
                    return (True if True in (a,b) else False if a is False and b is False else None) if op=='||' else (False if False in (a,b) else True if a is True and b is True else None)
        if s.startswith('!'):v=self.predicate(s[1:],env,control);return None if v is None else not v
        if re.fullmatch(r'\$\?\w+',s):return s[2:] in env
        if re.match(r'^-[edf]\s',s):
            path=self.value(s[3:],env)
            if path.unknown:return None
            try:self.files.resolve(control,path.text);return True
            except ValueError:return False
        compare=re.fullmatch(r'(.*?)\s*(==|!=|<=|>=|<|>)\s*(.*?)',s)
        if compare:
            a=self.value(compare[1],env);b=self.value(compare[3],env)
            if a.unknown or b.unknown:return None
            x,y=a.text,b.text;op=compare[2]
            if op in ('==','!='):return (x==y) if op=='==' else (x!=y)
            try:
                x,y=float(x),float(y)
                return {'<':x<y,'>':x>y,'<=':x<=y,'>=':x>=y}[op]
            except ValueError:return None
        return None

    def execute(self,nodes,env,control,conditional=False):
        for node in nodes:
            self.steps+=1
            if self.steps>self.limit:raise ValueError('控制声明分析步数超限，未无限求值')
            kind,line,*data=node
            if kind=='if':
                condition,yes,no=data;truth=self.predicate(condition,env,control)
                self.trace.append(dict(control=control,line=line,condition=condition,result=truth))
                if truth is not None:
                    signal=self.execute(yes if truth else no,env,control,conditional)
                    if signal:return signal
                else:
                    a,b=copy.copy(env),copy.copy(env)
                    sa=self.execute(yes,a,control,True);sb=self.execute(no,b,control,True)
                    for key in set(a)|set(b):
                        env[key]=a[key] if key in a and a.get(key)==b.get(key) else Value('${'+key+'}',frozenset({key,'unknown branch'}))
                    if sa==sb and sa:return sa
            elif kind=='loop':
                header,body=data
                if header.startswith('foreach'):
                    match=re.fullmatch(r'foreach\s+(\w+)\s*\((.*)\)',header);seq=self.value(match[2],env) if match else Value('',frozenset({'foreach'}))
                    if seq.unknown:
                        a=copy.copy(env);self.execute(body,a,control,True)
                        for key in set(a)|set(env):
                            if a.get(key)!=env.get(key):env[key]=Value('${'+key+'}',frozenset({key,'unknown loop'}))
                    else:
                        values=shlex.split(seq.text)
                        if len(values)>self.loop_limit:raise ValueError('foreach循环超限')
                        for value in values:
                            env[match[1]]=Value(value);signal=self.execute(body,env,control,conditional)
                            if signal=='exit':return signal
                            if signal=='break':break
                else:
                    condition=condition_head(header)[0]
                    for iteration in range(self.loop_limit+1):
                        truth=self.predicate(condition,env,control)
                        if truth is False:break
                        if truth is None:raise ValueError('输入相关while条件未知，不能伪造循环日期：'+condition)
                        if iteration==self.loop_limit:raise ValueError('while循环超限')
                        before=copy.copy(env);signal=self.execute(body,env,control,conditional)
                        if signal=='exit':return signal
                        if signal=='break':break
                        if env==before:raise ValueError('while无可验证进展')
            else:
                text=data[0]
                if text.split()[0]=='exit':return 'exit'
                if text=='break':return 'break'
                match=re.fullmatch(r'(?:set\s+(\w+)\s*=\s*|setenv\s+(\w+)\s*)(.*)',text)
                if match:
                    key=match[1] or match[2];env[key]=self.value(match[3],env)
                    self.trace.append(dict(control=control,line=line,variable=key,value=env[key].text,unknown=sorted(env[key].unknown),conditional=conditional))
                elif text.startswith('@ '):
                    match=re.fullmatch(r'@\s+(\w+)\s*=\s*(.*)',text)
                    if match:
                        v=self.expand(match[2],env)
                        try:env[match[1]]=Value(self.number(v.text)) if not v.unknown else v
                        except ValueError:env[match[1]]=Value('${'+match[1]+'}',frozenset({match[1]}))
                elif text.startswith(('unset ','unsetenv ')):
                    for key in text.split()[1:]:env.pop(key,None)
                elif text.startswith('source '):
                    path=self.value(text[7:].split()[0],env)
                    try:
                        if path.unknown:raise ValueError('source路径未知')
                        target,_=self.files.resolve(control,path.text)
                        if target in self.active or len(self.active)>=8:raise ValueError('source循环或深度超限')
                        with self.files.files[target][0]() as s:content=s.read().decode('utf-8-sig')
                        self.active.append(target);self.execute(parse(logical_lines(content)),env,target,conditional);self.active.pop()
                        self.includes.append(dict(control=control,line=line,target=target,status='PARSED_NOT_EXECUTED'))
                    except ValueError as e:
                        self.includes.append(dict(control=control,line=line,reference=path.text,status='UNKNOWN',reason=str(e)))
                        for key in list(env):env[key]=Value('${'+key+'}',frozenset({key,'unresolved source'}))
                elif text.startswith('eval '):
                    for key in list(env):env[key]=Value('${'+key+'}',frozenset({key,'unsupported eval'}))
                elif 'mpirun' in text or re.search(r'\$\{?EXEC\}?\b',text) and not text.startswith(('echo ','set ')):
                    self.invocation_seen=True
                    if not conditional:self.frames.append(dict(control=control,line=line,values=copy.copy(env)))
                # Filesystem, host commands and model invocations never execute.

    def analyze(self,control):
        with self.files.files[control][0]() as s:text=s.read().decode('utf-8-sig')
        env={};self.active=[control]
        signal=self.execute(parse(logical_lines(text)),env,control)
        frames=self.frames or ([] if signal or self.invocation_seen else [dict(control=control,line=None,values=env)])
        return dict(frames=frames,trace=self.trace,includes=self.includes,steps=self.steps,commands_executed=False)
