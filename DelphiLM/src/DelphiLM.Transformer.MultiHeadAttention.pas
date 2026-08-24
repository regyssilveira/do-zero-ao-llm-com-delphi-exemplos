{ Licensed under the Apache License, Version 2.0. See LICENSE. }
unit DelphiLM.Transformer.MultiHeadAttention;

interface

uses
  DelphiLM.Core.Random, DelphiLM.Math.Tensor,
  DelphiLM.Neural.Parameters, System.Generics.Collections;

type
  TMultiHeadCausalSelfAttention = class
  private
    FCachedConcatenated, FCachedInput: TTensor;
    FCachedKeys, FCachedQueries, FCachedValues, FCachedWeights: TArray<TTensor>;
    FEmbeddingDimension, FHeadCount, FHeadDimension: Integer;
    FKeyBias, FKeyWeights, FOutputBias, FOutputWeights,
    FQueryBias, FQueryWeights, FValueBias, FValueWeights: TTrainableParameter;
    procedure BackwardAttention(const AHead: Integer;
      const AGradOutput: TTensor; out AGradQueries, AGradKeys,
      AGradValues: TTensor);
    procedure BackwardProjection(const AHead: Integer;
      const AGradProjected: TTensor; const AWeights,
      ABias: TTrainableParameter; const AGradInput: TTensor);
    procedure ClearCache;
    procedure InitializeWeights(const AWeights: TTrainableParameter;
      var ARandom: TXorShift64Star);
    function ProjectHead(const AInput: TTensor; const AWeights,
      ABias: TTrainableParameter; const AHead: Integer): TTensor;
  public
    constructor Create(const AEmbeddingDimension, AHeadCount: Integer;
      var ARandom: TXorShift64Star);
    destructor Destroy; override;
    function Forward(const AInput: TTensor): TTensor;
    function Backward(const AGradOutput: TTensor): TTensor;
    procedure CollectParameters(const AList: TList<TTrainableParameter>);
    procedure ZeroGrad;
    procedure SetQueryWeight(const AOutput, AInput: Integer; const AValue: Single);
    procedure SetKeyWeight(const AOutput, AInput: Integer; const AValue: Single);
    procedure SetValueWeight(const AOutput, AInput: Integer; const AValue: Single);
    procedure SetOutputWeight(const AOutput, AInput: Integer; const AValue: Single);
    property EmbeddingDimension: Integer read FEmbeddingDimension;
    property HeadCount: Integer read FHeadCount;
    property HeadDimension: Integer read FHeadDimension;
  end;

implementation

uses
  DelphiLM.Transformer.Attention, System.Math, System.SysUtils;

function CloneMatrix(const ASource: TTensor): TTensor;
var I: Integer;
begin
  Result := TTensor.Create([ASource.Dimension(0), ASource.Dimension(1)]);
  for I := 0 to ASource.ElementCount - 1 do
    Result.SetFlatValue(I, ASource.FlatValue(I));
end;

constructor TMultiHeadCausalSelfAttention.Create(
  const AEmbeddingDimension, AHeadCount: Integer;
  var ARandom: TXorShift64Star);
begin
  inherited Create;
  if AEmbeddingDimension <= 0 then
    raise EDelphiLMTensorError.Create('EmbeddingDimension deve ser positivo.');
  if AHeadCount <= 0 then
    raise EDelphiLMTensorError.Create('HeadCount deve ser positivo.');
  if AEmbeddingDimension mod AHeadCount <> 0 then
    raise EDelphiLMTensorError.Create(
      'EmbeddingDimension deve ser divisível por HeadCount.');
  FEmbeddingDimension := AEmbeddingDimension;
  FHeadCount := AHeadCount;
  FHeadDimension := AEmbeddingDimension div AHeadCount;
  FQueryWeights := TTrainableParameter.Create([AEmbeddingDimension, AEmbeddingDimension]);
  FKeyWeights := TTrainableParameter.Create([AEmbeddingDimension, AEmbeddingDimension]);
  FValueWeights := TTrainableParameter.Create([AEmbeddingDimension, AEmbeddingDimension]);
  FOutputWeights := TTrainableParameter.Create([AEmbeddingDimension, AEmbeddingDimension]);
  FQueryBias := TTrainableParameter.Create([AEmbeddingDimension]);
  FKeyBias := TTrainableParameter.Create([AEmbeddingDimension]);
  FValueBias := TTrainableParameter.Create([AEmbeddingDimension]);
  FOutputBias := TTrainableParameter.Create([AEmbeddingDimension]);
  InitializeWeights(FQueryWeights, ARandom);
  InitializeWeights(FKeyWeights, ARandom);
  InitializeWeights(FValueWeights, ARandom);
  InitializeWeights(FOutputWeights, ARandom);
end;

destructor TMultiHeadCausalSelfAttention.Destroy;
begin
  ClearCache;
  FOutputBias.Free; FValueBias.Free; FKeyBias.Free; FQueryBias.Free;
  FOutputWeights.Free; FValueWeights.Free; FKeyWeights.Free; FQueryWeights.Free;
  inherited;
end;

procedure TMultiHeadCausalSelfAttention.ClearCache;
var H: Integer;
begin
  FreeAndNil(FCachedConcatenated); FreeAndNil(FCachedInput);
  for H := 0 to High(FCachedQueries) do
  begin
    FCachedWeights[H].Free; FCachedValues[H].Free;
    FCachedKeys[H].Free; FCachedQueries[H].Free;
  end;
  SetLength(FCachedQueries, 0); SetLength(FCachedKeys, 0);
  SetLength(FCachedValues, 0); SetLength(FCachedWeights, 0);
end;

procedure TMultiHeadCausalSelfAttention.InitializeWeights(
  const AWeights: TTrainableParameter; var ARandom: TXorShift64Star);
var I, O: Integer; Limit: Single;
begin
  Limit := Sqrt(6.0 / (2 * FEmbeddingDimension));
  for O := 0 to FEmbeddingDimension - 1 do
    for I := 0 to FEmbeddingDimension - 1 do
      AWeights.Value.SetValue((2 * ARandom.NextSingle - 1) * Limit, [O, I]);
end;

function TMultiHeadCausalSelfAttention.ProjectHead(const AInput: TTensor;
  const AWeights, ABias: TTrainableParameter; const AHead: Integer): TTensor;
var C, I, O, P: Integer; Sum: Single;
begin
  Result := TTensor.Create([AInput.Dimension(0), FHeadDimension]);
  for P := 0 to AInput.Dimension(0) - 1 do
    for C := 0 to FHeadDimension - 1 do
    begin
      O := AHead * FHeadDimension + C;
      Sum := ABias.Value.FlatValue(O);
      for I := 0 to FEmbeddingDimension - 1 do
        Sum := Sum + AWeights.Value.ValueAt([O, I]) * AInput.ValueAt([P, I]);
      Result.SetValue(Sum, [P, C]);
    end;
end;

function TMultiHeadCausalSelfAttention.Forward(const AInput: TTensor): TTensor;
var A: TTensor; C, H, I, P: Integer; Sum: Single;
begin
  if (AInput = nil) or (AInput.Rank <> 2) or
     (AInput.Dimension(1) <> FEmbeddingDimension) then
    raise EDelphiLMTensorError.CreateFmt(
      'Self-attention esperava shape [T,%d].', [FEmbeddingDimension]);
  ClearCache;
  FCachedInput := CloneMatrix(AInput);
  FCachedConcatenated := TTensor.Create([AInput.Dimension(0), FEmbeddingDimension]);
  SetLength(FCachedQueries, FHeadCount); SetLength(FCachedKeys, FHeadCount);
  SetLength(FCachedValues, FHeadCount); SetLength(FCachedWeights, FHeadCount);
  for H := 0 to FHeadCount - 1 do
  begin
    FCachedQueries[H] := ProjectHead(AInput, FQueryWeights, FQueryBias, H);
    FCachedKeys[H] := ProjectHead(AInput, FKeyWeights, FKeyBias, H);
    FCachedValues[H] := ProjectHead(AInput, FValueWeights, FValueBias, H);
    A := CausalScaledDotProductAttention(FCachedQueries[H], FCachedKeys[H],
      FCachedValues[H], FCachedWeights[H]);
    try
      for P := 0 to AInput.Dimension(0) - 1 do
        for C := 0 to FHeadDimension - 1 do
          FCachedConcatenated.SetValue(A.ValueAt([P, C]),
            [P, H * FHeadDimension + C]);
    finally A.Free end;
  end;
  Result := TTensor.Create([AInput.Dimension(0), FEmbeddingDimension]);
  for P := 0 to AInput.Dimension(0) - 1 do
    for C := 0 to FEmbeddingDimension - 1 do
    begin
      Sum := FOutputBias.Value.FlatValue(C);
      for I := 0 to FEmbeddingDimension - 1 do
        Sum := Sum + FOutputWeights.Value.ValueAt([C, I]) *
          FCachedConcatenated.ValueAt([P, I]);
      Result.SetValue(Sum, [P, C]);
    end;
end;

procedure TMultiHeadCausalSelfAttention.BackwardAttention(
  const AHead: Integer; const AGradOutput: TTensor;
  out AGradQueries, AGradKeys, AGradValues: TTensor);
var C, J, P: Integer; GW, GS, SD, Scale: Double;
begin
  AGradQueries := TTensor.Create([AGradOutput.Dimension(0), FHeadDimension]);
  AGradKeys := TTensor.Create([AGradOutput.Dimension(0), FHeadDimension]);
  AGradValues := TTensor.Create([AGradOutput.Dimension(0), FHeadDimension]);
  Scale := Sqrt(FHeadDimension);
  for P := 0 to AGradOutput.Dimension(0) - 1 do
  begin
    SD := 0;
    for J := 0 to P do
    begin
      GW := 0;
      for C := 0 to FHeadDimension - 1 do
      begin
        GW := GW + AGradOutput.ValueAt([P, C]) * FCachedValues[AHead].ValueAt([J, C]);
        AGradValues.SetValue(AGradValues.ValueAt([J, C]) +
          FCachedWeights[AHead].ValueAt([P, J]) * AGradOutput.ValueAt([P, C]), [J, C]);
      end;
      SD := SD + FCachedWeights[AHead].ValueAt([P, J]) * GW;
    end;
    for J := 0 to P do
    begin
      GW := 0;
      for C := 0 to FHeadDimension - 1 do
        GW := GW + AGradOutput.ValueAt([P, C]) * FCachedValues[AHead].ValueAt([J, C]);
      GS := FCachedWeights[AHead].ValueAt([P, J]) * (GW - SD);
      for C := 0 to FHeadDimension - 1 do
      begin
        AGradQueries.SetValue(AGradQueries.ValueAt([P, C]) +
          GS * FCachedKeys[AHead].ValueAt([J, C]) / Scale, [P, C]);
        AGradKeys.SetValue(AGradKeys.ValueAt([J, C]) +
          GS * FCachedQueries[AHead].ValueAt([P, C]) / Scale, [J, C]);
      end;
    end;
  end;
end;

procedure TMultiHeadCausalSelfAttention.BackwardProjection(
  const AHead: Integer; const AGradProjected: TTensor;
  const AWeights, ABias: TTrainableParameter; const AGradInput: TTensor);
var C, I, O, P: Integer; G: Single;
begin
  for P := 0 to AGradProjected.Dimension(0) - 1 do
    for C := 0 to FHeadDimension - 1 do
    begin
      O := AHead * FHeadDimension + C; G := AGradProjected.ValueAt([P, C]);
      ABias.AddGradientFlat(O, G);
      for I := 0 to FEmbeddingDimension - 1 do
      begin
        AWeights.Gradient.SetValue(AWeights.Gradient.ValueAt([O, I]) +
          G * FCachedInput.ValueAt([P, I]), [O, I]);
        AGradInput.SetValue(AGradInput.ValueAt([P, I]) +
          AWeights.Value.ValueAt([O, I]) * G, [P, I]);
      end;
    end;
end;

function TMultiHeadCausalSelfAttention.Backward(const AGradOutput: TTensor): TTensor;
var C, H, I, P: Integer; GC, GH, GQ, GK, GV: TTensor;
begin
  if FCachedInput = nil then
    raise EDelphiLMTensorError.Create('Backward exige um Forward anterior.');
  Result := TTensor.Create([FCachedInput.Dimension(0), FEmbeddingDimension]);
  GC := TTensor.Create([FCachedInput.Dimension(0), FEmbeddingDimension]);
  try
    for P := 0 to AGradOutput.Dimension(0) - 1 do
      for C := 0 to FEmbeddingDimension - 1 do
      begin
        FOutputBias.AddGradientFlat(C, AGradOutput.ValueAt([P, C]));
        for I := 0 to FEmbeddingDimension - 1 do
        begin
          FOutputWeights.Gradient.SetValue(FOutputWeights.Gradient.ValueAt([C, I]) +
            AGradOutput.ValueAt([P, C]) * FCachedConcatenated.ValueAt([P, I]), [C, I]);
          GC.SetValue(GC.ValueAt([P, I]) + FOutputWeights.Value.ValueAt([C, I]) *
            AGradOutput.ValueAt([P, C]), [P, I]);
        end;
      end;
    for H := 0 to FHeadCount - 1 do
    begin
      GH := TTensor.Create([FCachedInput.Dimension(0), FHeadDimension]);
      GQ := nil; GK := nil; GV := nil;
      try
        for P := 0 to FCachedInput.Dimension(0) - 1 do
          for C := 0 to FHeadDimension - 1 do
            GH.SetValue(GC.ValueAt([P, H * FHeadDimension + C]), [P, C]);
        BackwardAttention(H, GH, GQ, GK, GV);
        BackwardProjection(H, GQ, FQueryWeights, FQueryBias, Result);
        BackwardProjection(H, GK, FKeyWeights, FKeyBias, Result);
        BackwardProjection(H, GV, FValueWeights, FValueBias, Result);
      finally GV.Free; GK.Free; GQ.Free; GH.Free end;
    end;
  finally GC.Free end;
end;

procedure TMultiHeadCausalSelfAttention.CollectParameters(const AList: TList<TTrainableParameter>);
begin
  AList.Add(FQueryWeights); AList.Add(FQueryBias); AList.Add(FKeyWeights);
  AList.Add(FKeyBias); AList.Add(FValueWeights); AList.Add(FValueBias);
  AList.Add(FOutputWeights); AList.Add(FOutputBias);
end;

procedure TMultiHeadCausalSelfAttention.ZeroGrad;
begin
  FQueryWeights.ZeroGrad; FQueryBias.ZeroGrad; FKeyWeights.ZeroGrad;
  FKeyBias.ZeroGrad; FValueWeights.ZeroGrad; FValueBias.ZeroGrad;
  FOutputWeights.ZeroGrad; FOutputBias.ZeroGrad;
end;

procedure TMultiHeadCausalSelfAttention.SetQueryWeight(const AOutput, AInput: Integer; const AValue: Single);
begin FQueryWeights.Value.SetValue(AValue, [AOutput, AInput]) end;
procedure TMultiHeadCausalSelfAttention.SetKeyWeight(const AOutput, AInput: Integer; const AValue: Single);
begin FKeyWeights.Value.SetValue(AValue, [AOutput, AInput]) end;
procedure TMultiHeadCausalSelfAttention.SetValueWeight(const AOutput, AInput: Integer; const AValue: Single);
begin FValueWeights.Value.SetValue(AValue, [AOutput, AInput]) end;
procedure TMultiHeadCausalSelfAttention.SetOutputWeight(const AOutput, AInput: Integer; const AValue: Single);
begin FOutputWeights.Value.SetValue(AValue, [AOutput, AInput]) end;

end.
