function [theta, phi, psi] = control_sliders()
    % 创建一个figure窗口
    f = figure('Name', 'Angle Control', 'Position', [100, 100, 400, 300]);
    
    % 创建theta滑块
    thetaSlider = uicontrol('Style', 'slider', 'Min', 0, 'Max', 90, ...
                            'Position', [100, 220, 200, 20], ...
                            'Callback', @thetaSlider_callback);
    thetaLabel = uicontrol('Style', 'text', 'Position', [100, 240, 200, 20], ...
                           'String', 'Theta: 0');

    % 创建phi滑块
    phiSlider = uicontrol('Style', 'slider', 'Min', 0, 'Max', 90, ...
                          'Position', [100, 140, 200, 20], ...
                          'Callback', @phiSlider_callback);
    phiLabel = uicontrol('Style', 'text', 'Position', [100, 160, 200, 20], ...
                         'String', 'Phi: 0');

    % 创建psi滑块
    psiSlider = uicontrol('Style', 'slider', 'Min', 0, 'Max', 90, ...
                          'Position', [100, 60, 200, 20], ...
                          'Callback', @psiSlider_callback);
    psiLabel = uicontrol('Style', 'text', 'Position', [100, 80, 200, 20], ...
                         'String', 'Psi: 0');

    % 滑块回调函数
    function thetaSlider_callback(hObject, eventdata)
        theta = get(hObject, 'Value');
        set(thetaLabel, 'String', sprintf('Theta: %.2f', theta));
    end

    function phiSlider_callback(hObject, eventdata)
        phi = get(hObject, 'Value');
        set(phiLabel, 'String', sprintf('Phi: %.2f', phi));
    end

    function psiSlider_callback(hObject, eventdata)
        psi = get(hObject, 'Value');
        set(psiLabel, 'String', sprintf('Psi: %.2f', psi));
    end
end