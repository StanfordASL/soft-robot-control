fileIdx = ['1', '2', '3'];
modes = [];
for idx=1:size(fileIdx, 2)
    modeFile = sprintf('mode%s.csv', fileIdx(idx));
    mode = load(modeFile);
    mode = mode / norm(mode);
    
    modes = [modes, mode];
end

% Doing this leads to sligtly off-axis displacement
%modes = orthogonalizeGramSchmidt(modes);

for modeidx=1:size(modes,2)
    modeReshaped = [modes(1:3:end, modeidx) modes(2:3:end, modeidx) modes(3:3:end, modeidx)].';
    
    newModeFile = sprintf('mode%s.mat', fileIdx(modeidx));
    modeName = sprintf('mode%s', fileIdx(modeidx));
    S.(modeName) = modeReshaped;
    save(newModeFile, '-struct', 'S');
end