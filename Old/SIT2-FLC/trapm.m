function y = trapm(x,params,kk)
for m=1:kk;
    a = params(m,1); b = params(m,2); c = params(m,3); d = params(m,4);
    index = find(x >= b);
    if ~isempty(index),
        y1(index) = ones(size(index));
    end
    index = find(x < a);
    if ~isempty(index),
        y1(index) = zeros(size(index));
    end
    index = find(a <= x & x < b);
    if ~isempty(index) & a ~= b,
        y1(index) = (x(index)-a)/(b-a);
    end
    index = find(x <= c);
    if ~isempty(index),
        y2(index) = ones(size(index));
    end
    index = find(x > d);
    if ~isempty(index),
        y2(index) = zeros(size(index));
    end
    index = find(c < x & x <= d);
    if ~isempty(index) & c ~= d,
        y2(index) = (d-x(index))/(d-c);
    end

    y(m) = min(y1, y2);
end
