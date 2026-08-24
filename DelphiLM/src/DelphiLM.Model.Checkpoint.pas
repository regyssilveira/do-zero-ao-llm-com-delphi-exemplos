{ Licensed under the Apache License, Version 2.0. See LICENSE. }
unit DelphiLM.Model.Checkpoint;

interface

uses
  DelphiLM.Model.LanguageModel;

procedure SaveCheckpoint(const AModel: TDelphiLanguageModel;
  const AFileName: string);
procedure LoadCheckpoint(const AModel: TDelphiLanguageModel;
  const AFileName: string);

implementation

uses
  DelphiLM.Core.Config, DelphiLM.Math.Tensor, DelphiLM.Neural.Parameters,
  System.Classes, System.Generics.Collections, System.SysUtils;

const
  CheckpointMagic: UInt64 = $444C4D43484B5031;
  CheckpointVersion: Cardinal = 1;

procedure WriteInteger(const S: TStream; const V: Integer);
begin S.WriteBuffer(V, SizeOf(V)) end;
function ReadInteger(const S: TStream): Integer;
begin S.ReadBuffer(Result, SizeOf(Result)) end;

procedure WriteConfig(const S: TStream; const C: TDelphiLMConfig);
begin
  WriteInteger(S, C.VocabularySize); WriteInteger(S, C.ContextLength);
  WriteInteger(S, C.EmbeddingDimension); WriteInteger(S, C.BlockCount);
  WriteInteger(S, C.AttentionHeadCount); WriteInteger(S, C.FeedForwardDimension);
end;

procedure ValidateConfig(const S: TStream; const C: TDelphiLMConfig);
begin
  if (ReadInteger(S) <> C.VocabularySize) or
     (ReadInteger(S) <> C.ContextLength) or
     (ReadInteger(S) <> C.EmbeddingDimension) or
     (ReadInteger(S) <> C.BlockCount) or
     (ReadInteger(S) <> C.AttentionHeadCount) or
     (ReadInteger(S) <> C.FeedForwardDimension) then
    raise EDelphiLMTensorError.Create('Checkpoint incompatível com a configuração do modelo.');
end;

procedure SaveCheckpoint(const AModel: TDelphiLanguageModel; const AFileName: string);
var Count, I, P: Integer; Parameters: TList<TTrainableParameter>;
  S: TFileStream; Value: Single; Magic: UInt64; Version: Cardinal;
begin
  if AModel = nil then raise EArgumentNilException.Create('AModel');
  Parameters := TList<TTrainableParameter>.Create;
  S := TFileStream.Create(AFileName, fmCreate);
  try
    Magic := CheckpointMagic; Version := CheckpointVersion;
    S.WriteBuffer(Magic, SizeOf(Magic)); S.WriteBuffer(Version, SizeOf(Version));
    WriteConfig(S, AModel.Config); AModel.CollectParameters(Parameters);
    Count := Parameters.Count; S.WriteBuffer(Count, SizeOf(Count));
    for P := 0 to Parameters.Count - 1 do
    begin
      Count := Parameters[P].Value.ElementCount; S.WriteBuffer(Count, SizeOf(Count));
      for I := 0 to Count - 1 do
      begin Value := Parameters[P].Value.FlatValue(I); S.WriteBuffer(Value, SizeOf(Value)) end;
    end;
  finally S.Free; Parameters.Free end;
end;

procedure LoadCheckpoint(const AModel: TDelphiLanguageModel; const AFileName: string);
var Count, Expected, I, P: Integer; Parameters: TList<TTrainableParameter>;
  S: TFileStream; Value: Single; Magic: UInt64; Version: Cardinal;
begin
  if AModel = nil then raise EArgumentNilException.Create('AModel');
  Parameters := TList<TTrainableParameter>.Create;
  S := TFileStream.Create(AFileName, fmOpenRead or fmShareDenyWrite);
  try
    S.ReadBuffer(Magic, SizeOf(Magic)); S.ReadBuffer(Version, SizeOf(Version));
    if Magic <> CheckpointMagic then raise EDelphiLMTensorError.Create('Arquivo não é um checkpoint DelphiLM.');
    if Version <> CheckpointVersion then raise EDelphiLMTensorError.CreateFmt(
      'Versão de checkpoint %d não suportada.', [Version]);
    ValidateConfig(S, AModel.Config); AModel.CollectParameters(Parameters);
    S.ReadBuffer(Count, SizeOf(Count));
    if Count <> Parameters.Count then raise EDelphiLMTensorError.Create('Quantidade de parâmetros incompatível.');
    for P := 0 to Parameters.Count - 1 do
    begin
      S.ReadBuffer(Expected, SizeOf(Expected));
      if Expected <> Parameters[P].Value.ElementCount then
        raise EDelphiLMTensorError.Create('Shape de parâmetro incompatível.');
      for I := 0 to Expected - 1 do
      begin S.ReadBuffer(Value, SizeOf(Value)); Parameters[P].Value.SetFlatValue(I, Value) end;
    end;
    if S.Position <> S.Size then raise EDelphiLMTensorError.Create('Checkpoint contém dados excedentes.');
  finally S.Free; Parameters.Free end;
end;

end.
