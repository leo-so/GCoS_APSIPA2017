% Input:
% x: input signal in column (single channel)
% fr: desired frequency resolution (Hz)
% fs sampling frequency of the input signal (Hz)
% Hop: hop isze (sample)
% h: window function
% g1: the first-layer gamma
% g2: the second-layer gamma (in our case g3 = 1)
% Example: CFP_GCoS(x, 4, 44100, 441, hamming(4097), 0.6, 0.8)
% Output: 
% tfr: spectrogram
% ceps: generalized cepstrum
% GCoS: generalized cepstrum of spectrum
% upcp: pitch profile of GCoS (piano roll)
% upcpt: pitch profile of cepstrum (piano roll)
% upcp_final:pitch detection (piano roll)
% t:time index
% Reference:
% Li Su, "Between Homomorphic Signal Processing and Deep Neural Networks: 
% Constructing Deep Algorithms for Polyphonic Music Transcription," Proc. 
% Asia Pacific Signal and Infor. Proc. Asso. Annual Summit and Conf. 
% (APSIPA ASC), December 2017.
function [tfr, ceps, GCoS, upcp, upcpt, upcp_final, t] = CFP_GCoS(x, fr, fs, Hop, h, g1, g2)
fc = 20; tc=1/20000;
[tfr, f, t, N] = STFT(x, fr, fs, Hop, h);
tfr = abs(tfr).^g1;

fc_idx = round(fc/fr);
tc_idx = round(fs*tc);

tfr = nonlinear_func(abs(tfr), g1, fc_idx);
ceps = real(fft(tfr))./sqrt(N);
ceps = nonlinear_func(ceps, g2, tc_idx);

GCoS = real(fft(ceps))./sqrt(N);
GCoS = nonlinear_func(GCoS, 1, fc_idx);

tfr = tfr(1:round(N/2),:);
ceps = ceps(1:round(N/2),:);
GCoS = GCoS(1:round(N/2),:);

HighFreqIdx = round((1/tc)/fr)+1;
f = f(1:HighFreqIdx);
tfr = tfr(1:HighFreqIdx,:);
HighQuefIdx = round(fs/fc)+1;
q = (0:HighQuefIdx-1)./fs;
ceps = ceps(1:HighQuefIdx,:);

GCoS = PeakPicking(GCoS);
ceps = PeakPicking(ceps);

% tceps = CepsConvertFreq(ceps, f, fs);

midi_num=-3:133;
fd=440*2.^((midi_num-69-0.5)/12);
upcp = PitchProfileFreq(GCoS, f, fd);
upcpt = PitchProfileQuef(ceps, 1./q, fd);

[upcp, upcpt, upcp_final] = PitchFusion(upcp, upcpt, 4, 4, 0.7, 12);
end

function [tfr, f, t, N] = STFT(x, fr, fs, Hop, h)
% STFT by Li Su, 2017
% Modified from: Time-frequency toolbox by Patrick Flanderin
% http://tftb.nongnu.org/
% fr: desired frequency resolution
% fs: sampling frequency
% Note: fr = alpha * fs ->
% alpha = fr / fs
% tfr: optput STFT (full frequency)
% f: output frequency grid (only positive frequency) 
% t: output time grid (from the start to the end)

if size(h,2) > size(h,1)
    h = h';
end

	% for tfr
alpha = fr/fs;
N = length(-0.5+alpha:alpha:0.5);
Win_length = max(size(h));
f = fs.*linspace(0, 0.5, round(N/2))' ;

Lh = floor((Win_length-1)/2);
t = Hop:Hop:floor(length(x)/Hop)*Hop;
x_Frame = zeros(N, length(t));
for ii = 1:length(t)
    ti = t(ii); 
    tau = -min([round(N/2)-1,Lh,ti-1]):min([round(N/2)-1,Lh,length(x)-ti]);
    indices= rem(N+tau,N)+1;
    norm_h=norm(h(Lh+1+tau)); 

    x_Frame(indices,ii) = (x(ti+tau)-mean(x(ti+tau))).*conj( h(Lh+1+tau)) /norm_h;
end

tfr = fft(x_Frame, N, 1);
end

function X = nonlinear_func(X, g, cutoff)
if g~=0
    X(X<0) = 0;
    X(1:cutoff, :) = 0;
    X(end-cutoff+1:end, :) = 0;
    X = X.^g;
else
    X = log(X);
    X(1:cutoff, :) = 0;
    X(end-cutoff+1:end, :) = 0;
end
end

function upcp = PitchProfileFreq(spec, f, fd)

% freq_scale=(0:size(spec,1)-1).*fr;

spec=abs(spec);
upcp=zeros(136,size(spec,2));
for ii=25:136
    p_index = find(f > fd(ii) & f < fd(ii+1));
    if ~isempty(p_index)
    upcp(ii,:)=max(spec(p_index,:),[],1);
    end
end
upcp(isnan(upcp)|isinf(upcp))=0;
end

function upcp = PitchProfileQuef(acr, f, fd)

% freq_scale = fs./(1:size(acr,1)-1);

acr=abs(acr);
upcp=zeros(136,size(acr,2));
for ii=1:112
    p_index = find(f > fd(ii) & f < fd(ii+1));
    if ~isempty(p_index)
    upcp(ii,:)=max(acr(p_index,:),[],1);
    end
end
upcp(isnan(upcp)|isinf(upcp))=0;
end

function [upcp, upcpa, upcp_final] = PitchFusion(upcp, upcpa, nh, nr, ratio, NumPerOctave)
upcp_final = zeros(size(upcp));
A = 2*NumPerOctave+1;
B = size(upcp,1)-2*NumPerOctave;
for xi = 1:size(upcp, 2)
    for fi = A:B
        if IsCandidate(upcp(:,xi),upcpa(:,xi),fi,nh,nr,ratio, NumPerOctave)==1
            upcp_final(fi,xi)=1;
        end
    end
end
upcp = upcp(A:B,:);
upcpa = upcpa(A:B,:);
upcp_final = upcp_final(A:B,:);
end

function isornot = IsCandidate(s, c, pitch, num_s, num_c, ratio, NumPerOctave)
isornot=0; 
har=[0 12 19 24 28 31 34 36 38 40].*NumPerOctave./12;
if (min(s(pitch+har(1:num_s)))>0 && min(c(pitch-har(1:num_c)))>0) ...
        && ~(nnz(s(pitch:pitch+har(num_s)))>=ratio*(har(num_s)+1) && nnz(c(pitch:-1:pitch-har(num_c)))>=ratio*(har(num_c)+1))
    isornot=1; 
end
end

function Y = PeakPicking(X)
pre = X-[X(2:end,:); zeros(1, size(X,2))]; pre(pre<0)=0; pre(pre>0)=1;
post = X-[zeros(1, size(X,2)); X(1:end-1,:)]; post(post<0)=0; post(post>0)=1;
mm = pre.*post; %mm(mm<0)=0;
Y = X.*mm;
end
