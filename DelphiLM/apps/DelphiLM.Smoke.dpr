{ Licensed under the Apache License, Version 2.0. See LICENSE. }
program DelphiLM_Smoke;

{$APPTYPE CONSOLE}

uses
  System.SysUtils,
  DelphiLM.Core.Config in '..\src\DelphiLM.Core.Config.pas',
  DelphiLM.Core.Random in '..\src\DelphiLM.Core.Random.pas';

const
  Banner = '''
    DelphiLM smoke test
    Plataforma: Delphi 13 Florence / Win64
    ''';

var
  Random: TXorShift64Star;
begin
  try
    var Config := TDelphiLMConfig.Default;
    Config.Validate;
    Random.Initialize(Config.Seed);

    Writeln(Banner);
    Writeln('Parâmetros de referência: ', Config.ParameterCount);
    Writeln('Amostra determinística: ', FormatFloat('0.000000', Random.NextSingle));
  except
    on E: Exception do
    begin
      Writeln(ErrOutput, E.ClassName, ': ', E.Message);
      ExitCode := 1;
    end;
  end;
end.
