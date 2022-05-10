function inputRegressor = prepareRegressor(y,u,regressor) 
[ly,ny] = size(y); [lu,nu] = size(u);

if ny == 1
    OutName{1} = 'y';
else
    for k=1:ny
        OutName{k} = 'y' + string(k);
    end
end

if nu == 1
    InName{1} = 'u';
else
    for k=1:ny
        InName{k} = 'u' + string(k);
    end
end
reg = split(regressor,',');
Delays_y = []; Delays_u = [];
n = length(reg);

% Handle output variables
i=1;
for k = 1:ny
    for j = 1:n
        str = reg{j};
        if (sum(double(ismember(str,'y'))) > 0)
            [pos, del, msg] = FindVar([OutName{k} '(' 'k'], str);
            if ~isempty(msg)
                return
            end
            unidel = unique(del);
            if ~isempty(pos)
%                 str_y{i} = VarRep('y', pos, del, k, str);
%                 str_y{i} = str;
                Delays_y = [Delays_y, unidel];
                i=i+1;
            end
        end
    end
                Delays_y = sort(Delays_y);
end

% Handle input variables.
i=1;
for k = 1:nu
    for j = 1:n
        str = reg{j};
        if (sum(double(ismember(str,'u'))) > 0)
            [pos, del, msg] = FindVar([InName{k} '(' 'k'], str);
            if ~isempty(msg)
                return
            end
            unidel = unique(del);
            if ~isempty(pos)
%                 str_u{i} = VarRep('u', pos, del, k, str);
%                 str_u{i} = str;
                Delays_u = [Delays_u, unidel];
                i=i+1;
            end
        end
    end
                Delays_u = sort(Delays_u);
end

if length(Delays_y)==1
    y_k = [zeros(Delays_y(1),1); y(1:ly-Delays_y(1))];
    str_y{1} = 'y(k-'+string(Delays_y(1))+')';
else
    y_k = [zeros(Delays_y(1),1); y(1:ly-Delays_y(1))];
    str_y{1} = 'y(k-'+string(Delays_y(1))+')';
for k=2:length(Delays_y)
    y_k = [y_k, [zeros(Delays_y(k),1); y(1:ly-Delays_y(k))]];
    str_y{k} = 'y(k-'+string(Delays_y(k))+')';
end
end

if length(Delays_u)==1
    u_k = [zeros(Delays_u(1),1); u(1:lu-Delays_u(1))];
    str_u{1} = 'u(k-'+string(Delays_u(1))+')';
else
    u_k = [zeros(Delays_u(1),1); u(1:lu-Delays_u(1))];
    str_u{1} = 'u(k-'+string(Delays_u(1))+')';
for k=2:length(Delays_u)
    u_k = [u_k, [zeros(Delays_u(k),1); u(1:lu-Delays_u(k))]];
    str_u{k} = 'u(k-'+string(Delays_u(k))+')';
end
end

inputRegressor{1} = [y_k u_k];
inputRegressor{2} = [str_y str_u];

function [pos, del, msg] = FindVar(name, reg)

% If reg has fewer characters than name, then return [].
pos = [];
del = [];
msg = struct([]);
if (length(reg) < length(name))
    return;
end

% Get all positions of name in reg.
apos = findstr(name, reg);

if isempty(apos)
    return;
end

% Remove false name hits by looking at the beginning of the expression.
spos = [];
for i = 1:length(apos)
    if (apos(i) == 1)
        spos = [spos apos(i)];
    elseif ~isalpha_num(reg(apos(i)-1)) && ~isalpha_num(reg(apos(i)+length(name)))
        spos = [spos apos(i)];
    end
end
if isempty(spos)
    return
end

% Compute the delay del.
lpos = findstr(reg, ')');
for i = 1:length(spos)
    apos = lpos(lpos >= spos(i)+length(name));
    delstr = reg(spos(i)+length(name):apos(1)-1);
    if isempty(delstr)
        del = [del 0];
    else
        deltmp = -str2double(reg(spos(i)+length(name):apos(1)-1));
        if isnan(deltmp)
            msg = ctrlMsgUtils.message('Ident:utility:str2CustomRegDelayVal');
            msg = struct('identifier','Ident:utility:str2CustomRegDelayVal','message',msg);
            return
        else
            del = [del deltmp];
        end
    end
end

% Return spos as pos.
pos = spos;

function reg = VarRep(io, pos, del, num, reg)

for i = length(pos):-1:1
    lpos = findstr(reg, ')');
    lpos = lpos(lpos > pos(i));
    %k = length(pos)-i+1;
    reg = [reg(1:pos(i)-1), io, num2str(num,5), '_' num2str(del(i),5), reg(lpos(1)+1:end)];
end








