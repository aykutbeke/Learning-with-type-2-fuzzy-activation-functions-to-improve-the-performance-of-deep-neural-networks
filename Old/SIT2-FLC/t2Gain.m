function K = t2Gain(sigma,alpha,B1)
    if sigma>0
    K = 0.5*(B1/(sigma - sigma*alpha + alpha)...
        - (B1-B1*alpha)/(sigma*alpha-1));
    else
%     K = 0.5*(B1/(-sigma + sigma*alpha + alpha)...
%         - (B1-B1*alpha)/(-sigma*alpha-1));
     K = 0.5*(B1/(-sigma + sigma*alpha + alpha)...
        - (B1-B1*alpha)/(-sigma*alpha-1));
    
%      K = (0.5*(B1/(sigma - sigma*alpha + alpha)...
%         - (B1-B1*alpha)/(sigma*alpha-1)));
    end
end